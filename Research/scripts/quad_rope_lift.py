"""quad_rope_lift.py

Minimal Drake example: a quadcopter lifts a load using a rope that can go slack.

Model:
- Quad: rigid body constrained to vertical motion (prismatic joint to world).
- Load: free rigid body (sphere) with gravity + ground contact.
- Rope: tension-only spring-damper between attachment points on quad and load.

Rope law:
- If distance $d \le L_0$: tension $T=0$ (slack)
- If $d > L_0$: $T = k(d-L_0) + c\,\max(\dot d, 0)$ (taut, pulling only)

Visualization: Meshcat.
"""

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AbstractValue,
    BasicVector,
    Box,
    CoulombFriction,
    DiagramBuilder,
    ExternallyAppliedSpatialForce,
    HalfSpace,
    LeafSystem,
    MeshcatVisualizer,
    Parser,
    PrismaticJoint,
    Rgba,
    RigidTransform,
    Simulator,
    SpatialForce,
    SpatialVelocity,
    SpatialInertia,
    Sphere,
    StartMeshcat,
    UnitInertia,
    ContactModel,
    LogVectorOutput,
)


class QuadXZController(LeafSystem):
    """PD + feedforward controller for the quad's planar (x,z) motion.

    The quad is modeled with two prismatic joints in series:
    - world --(x joint)--> carriage --(z joint)--> quad

    Trajectory (time-based): lift to ``z_final`` first, then move left, then right.

    Input port: plant state ``x = [q; v]``.
    Output port: 2D actuation ``u = [u_x, u_z]``.
    """
    def __init__(
        self,
        plant,
        x_joint: PrismaticJoint,
        z_joint: PrismaticJoint,
        x0: float,
        x_left: float,
        x_right: float,
        z0: float,
        z_final: float,
        ramp_start: float,
        climb_rate: float,
        move_pause: float,
        move_rate: float,
        kp: float,
        kd: float,
        u_max_x: float,
        u_max_z: float,
        u_ff_z: float,
    ):
        super().__init__()
        self._plant = plant
        self._x_joint = x_joint
        self._z_joint = z_joint
        self._ctx = plant.CreateDefaultContext()

        self._n = plant.num_positions() + plant.num_velocities()
        self.DeclareVectorInputPort("x", BasicVector(self._n))
        self.DeclareVectorOutputPort("u", BasicVector(2), self._calc_u)

        self._x0 = float(x0)
        self._x_left = float(x_left)
        self._x_right = float(x_right)
        self._z0 = z0
        self._z_final = z_final
        self._t0 = ramp_start
        self._vz = climb_rate
        self._pause = float(move_pause)
        self._vx = float(move_rate)
        self._kp = kp
        self._kd = kd
        self._u_max_x = float(u_max_x)
        self._u_max_z = float(u_max_z)
        self._u_ff_z = float(u_ff_z)

        # Derived timing.
        climb_time = 0.0 if self._vz <= 0.0 else max((self._z_final - self._z0) / self._vz, 0.0)
        self._t_lift_done = self._t0 + climb_time

    @staticmethod
    def _ramp(a0: float, a1: float, t: float, t0: float, v: float):
        """Constant-speed ramp from a0 toward a1 starting at t0."""
        if t <= t0:
            return a0, 0.0
        if v <= 0.0:
            return a0, 0.0
        direction = np.sign(a1 - a0)
        if direction == 0.0:
            return a0, 0.0
        duration = abs(a1 - a0) / v
        if t >= t0 + duration:
            return a1, 0.0
        return a0 + direction * v * (t - t0), direction * v

    def _desired(self, t: float):
        # z: lift, then hold.
        z_des, zd_des = self._ramp(self._z0, self._z_final, t, self._t0, self._vz)

        # x: after lift completes, pause, go left, pause, go right.
        t1 = self._t_lift_done + self._pause
        x1, xd1 = self._ramp(self._x0, self._x_left, t, t1, self._vx)
        if t <= t1 or (x1 != self._x_left and xd1 != 0.0):
            return x1, xd1, z_des, zd_des

        t2 = t1 + (abs(self._x_left - self._x0) / self._vx if self._vx > 0 else 0.0) + self._pause
        x2, xd2 = self._ramp(self._x_left, self._x_right, t, t2, self._vx)
        return x2, xd2, z_des, zd_des

    def _calc_u(self, context, output):
        x = self.get_input_port(0).Eval(context)
        self._plant.SetPositionsAndVelocities(self._ctx, x)

        t = context.get_time()
        x_des, xd_des, z_des, zd_des = self._desired(t)

        x_q = self._x_joint.get_translation(self._ctx)
        x_v = self._x_joint.get_translation_rate(self._ctx)
        z_q = self._z_joint.get_translation(self._ctx)
        z_v = self._z_joint.get_translation_rate(self._ctx)

        u_x = self._kp * (x_des - x_q) + self._kd * (xd_des - x_v)
        u_z = self._kp * (z_des - z_q) + self._kd * (zd_des - z_v) + self._u_ff_z

        u_x = float(np.clip(u_x, -self._u_max_x, self._u_max_x))
        u_z = float(np.clip(u_z, -self._u_max_z, self._u_max_z))
        output.SetAtIndex(0, u_x)
        output.SetAtIndex(1, u_z)

    def set_u_ff_z(self, u_ff_z: float) -> None:
        self._u_ff_z = float(u_ff_z)


class TensionOnlyRope(LeafSystem):
    """Tension-only rope between two body-fixed attachment points.

    Input port: plant state ``x = [q; v]``.
    Output ports:
    - ``spatial_forces`` (abstract): list[ExternallyAppliedSpatialForce] (quad + load)
    - ``tension`` (vector): scalar tension $T$ (useful for logging)
    """
    def __init__(
        self,
        plant,
        quad_body,
        load_body,
        p_QoQq_Q: np.ndarray,  # attachment point in quad body frame
        p_LoLq_L: np.ndarray,  # attachment point in load body frame
        L0: float,             # nominal rope length
        k: float,              # spring stiffness (N/m)
        c: float,              # damping (N/(m/s))
        eps: float = 1e-9
    ):
        super().__init__()
        self._plant = plant
        self._ctx = plant.CreateDefaultContext()

        self._quad = quad_body
        self._load = load_body

        self._pQ = p_QoQq_Q.reshape(3,)
        self._pL = p_LoLq_L.reshape(3,)

        self._L0 = float(L0)
        self._k = float(k)
        self._c = float(c)
        self._eps = float(eps)

        self._n = plant.num_positions() + plant.num_velocities()
        self.DeclareVectorInputPort("x", BasicVector(self._n))

        def _alloc_forces():
            return AbstractValue.Make(
                [ExternallyAppliedSpatialForce(), ExternallyAppliedSpatialForce()]
            )

        self.DeclareAbstractOutputPort("spatial_forces", _alloc_forces, self._calc_forces)
        self.DeclareVectorOutputPort("tension", BasicVector(1), self._calc_tension)

        self._last_tension = 0.0

    def _kinematics(self, ctx):
        # Body poses in world.
        X_WQ = self._plant.EvalBodyPoseInWorld(ctx, self._quad)
        X_WL = self._plant.EvalBodyPoseInWorld(ctx, self._load)

        p_WQq = X_WQ.multiply(self._pQ)
        p_WLq = X_WL.multiply(self._pL)

        # Attachment point translational velocities from body spatial velocities.
        V_WQ = self._plant.EvalBodySpatialVelocityInWorld(ctx, self._quad)
        V_WL = self._plant.EvalBodySpatialVelocityInWorld(ctx, self._load)

        # r from body origin to attachment point, expressed in world:
        r_WQ = X_WQ.rotation().multiply(self._pQ)
        r_WL = X_WL.rotation().multiply(self._pL)

        v_WQq = V_WQ.translational() + np.cross(V_WQ.rotational(), r_WQ)
        v_WLq = V_WL.translational() + np.cross(V_WL.rotational(), r_WL)

        return p_WQq, p_WLq, v_WQq, v_WLq

    def _compute_tension_and_direction(self, ctx):
        p_WQq, p_WLq, v_WQq, v_WLq = self._kinematics(ctx)

        r = p_WQq - p_WLq
        d = float(np.linalg.norm(r))
        if d < self._eps:
            return 0.0, np.zeros(3)

        e = r / d  # direction from load attachment toward quad attachment
        stretch = d - self._L0

        if stretch <= 0.0:
            return 0.0, e

        stretch_rate = float(e.dot(v_WQq - v_WLq))
        # Pulling-only damping (do not "push" when shortening)
        T = self._k * stretch + self._c * max(stretch_rate, 0.0)
        return float(max(T, 0.0)), e

    def _calc_forces(self, context, output):
        x = self.get_input_port(0).Eval(context)
        self._plant.SetPositionsAndVelocities(self._ctx, x)

        T, e = self._compute_tension_and_direction(self._ctx)
        self._last_tension = T

        f_load_W = T * e          # force applied to load, in world frame
        f_quad_W = -f_load_W      # equal and opposite on quad

        forces = []

        # Force on quad
        fq = ExternallyAppliedSpatialForce()
        fq.body_index = self._quad.index()
        fq.p_BoBq_B = self._pQ
        fq.F_Bq_W = SpatialForce(tau=np.zeros(3), f=f_quad_W)
        forces.append(fq)

        # Force on load
        fl = ExternallyAppliedSpatialForce()
        fl.body_index = self._load.index()
        fl.p_BoBq_B = self._pL
        fl.F_Bq_W = SpatialForce(tau=np.zeros(3), f=f_load_W)
        forces.append(fl)

        output.set_value(forces)

    def _calc_tension(self, context, output):
        output.SetAtIndex(0, self._last_tension)


class RopeLineVisualizer(LeafSystem):
    """Draws a rope as a Meshcat line segment between two moving points."""

    def __init__(
        self,
        plant,
        quad_body,
        load_body,
        p_QoQq_Q: np.ndarray,
        p_LoLq_L: np.ndarray,
        meshcat,
        path: str = "rope",
        line_width: float = 4.0,
        slack_color: Rgba = Rgba(0.7, 0.7, 0.7, 1.0),
        taut_color: Rgba = Rgba(0.95, 0.2, 0.2, 1.0),
        L0: float | None = None,
        eps: float = 1e-9,
        publish_period: float = 1.0 / 60.0,
    ):
        super().__init__()
        self._plant = plant
        self._ctx = plant.CreateDefaultContext()
        self._quad = quad_body
        self._load = load_body

        self._pQ = p_QoQq_Q.reshape(3,)
        self._pL = p_LoLq_L.reshape(3,)

        self._meshcat = meshcat
        self._path = path
        self._line_width = float(line_width)
        self._slack = slack_color
        self._taut = taut_color
        self._L0 = None if L0 is None else float(L0)
        self._eps = float(eps)

        n = plant.num_positions() + plant.num_velocities()
        self.DeclareVectorInputPort("x", BasicVector(n))
        self.DeclarePeriodicPublishEvent(publish_period, 0.0, self._publish)

    def _attachment_points_W(self, ctx):
        X_WQ = self._plant.EvalBodyPoseInWorld(ctx, self._quad)
        X_WL = self._plant.EvalBodyPoseInWorld(ctx, self._load)
        return X_WQ.multiply(self._pQ), X_WL.multiply(self._pL)

    def _publish(self, context):
        x = self.get_input_port(0).Eval(context)
        self._plant.SetPositionsAndVelocities(self._ctx, x)

        p_WQq, p_WLq = self._attachment_points_W(self._ctx)
        pts = np.column_stack([p_WQq, p_WLq])

        rgba = self._taut
        if self._L0 is not None:
            d = float(np.linalg.norm(p_WQq - p_WLq))
            if d <= self._L0 + self._eps:
                rgba = self._slack

        self._meshcat.SetLine(self._path, pts, self._line_width, rgba)


def main():
    enable_recording = True
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    # Make contact easy: use point contact (no hydroelastic properties required).
    plant.set_contact_model(ContactModel.kPoint)

    g = 9.81

    quad_mass = 1.5     # kg
    quad_size = np.array([0.30, 0.30, 0.10])  # box dims (m)

    load_mass = 2.0     # kg
    load_radius = 0.12  # m

    rope_L0 = 2.0       # m (nominal length)
    rope_k = 800.0      # N/m (stiff -> near-inextensible when taut)
    rope_c = 40.0       # N/(m/s)

    # Initial conditions: load on ground, rope slack.
    x_quad0 = 0.0
    z_quad0 = 1.0
    z_quad_final = 3.0

    # Motion plan: lift, then move left, then right.
    ramp_start = 0.5
    climb_rate = 0.4  # m/s
    x_left = -1.0
    x_right = 1.0
    move_pause = 0.5  # s
    move_rate = 0.5   # m/s

    # Ensure the sim runs long enough to finish the full plan.
    lift_time = max((z_quad_final - z_quad0) / max(climb_rate, 1e-9), 0.0)
    left_time = abs(x_left - x_quad0) / max(move_rate, 1e-9)
    right_time = abs(x_right - x_left) / max(move_rate, 1e-9)
    sim_duration = ramp_start + lift_time + move_pause + left_time + move_pause + right_time + 0.5

    Iq = SpatialInertia(
        mass=quad_mass,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia.SolidBox(*quad_size),
    )
    quad = plant.AddRigidBody("quad", Iq)

    # Two prismatic joints in series: world -> x carriage -> z -> quad.
    carriage_inertia = SpatialInertia(
        mass=1e-3,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia.SolidBox(1e-3, 1e-3, 1e-3),
    )
    carriage = plant.AddRigidBody("carriage", carriage_inertia)

    x_joint = plant.AddJoint(
        PrismaticJoint(
            name="quad_x",
            frame_on_parent=plant.world_frame(),
            frame_on_child=carriage.body_frame(),
            axis=np.array([1.0, 0.0, 0.0]),
        )
    )
    z_joint = plant.AddJoint(
        PrismaticJoint(
            name="quad_z",
            frame_on_parent=carriage.body_frame(),
            frame_on_child=quad.body_frame(),
            axis=np.array([0.0, 0.0, 1.0]),
        )
    )
    plant.AddJointActuator("quad_x_act", x_joint)
    plant.AddJointActuator("quad_z_act", z_joint)

    Il = SpatialInertia(
        mass=load_mass,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia.SolidSphere(load_radius),
    )
    load = plant.AddRigidBody("load", Il)

    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.7)

    plant.RegisterCollisionGeometry(
        load, RigidTransform(), Sphere(load_radius), "load_collision", friction
    )
    plant.RegisterVisualGeometry(
        load, RigidTransform(), Sphere(load_radius), "load_visual", np.array([0.8, 0.2, 0.2, 1.0])
    )

    X_WG = RigidTransform()
    plant.RegisterCollisionGeometry(
        plant.world_body(), X_WG, HalfSpace(), "ground_collision", friction
    )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        RigidTransform([0.0, 0.0, -0.02]),
        Box(10.0, 10.0, 0.04),
        "ground_visual",
        np.array([0.7, 0.7, 0.7, 1.0]),
    )

    # Visuals: weld Drake's quadrotor URDF model onto our simplified dynamic body.
    # This keeps the x/z prismatic-joint dynamics but replaces the cube visuals.
    parser = Parser(plant)
    (quad_visual_instance,) = parser.AddModels(
        url="package://drake_models/skydio_2/quadrotor.urdf"
    )
    quad_visual_base = plant.GetBodyByName("base_link", quad_visual_instance)
    plant.WeldFrames(quad.body_frame(), quad_visual_base.body_frame(), RigidTransform())

    # Finalize the plant (no topology changes allowed after this).
    plant.Finalize()

    p_QoQq_Q = np.array([0.0, 0.0, -quad_size[2] / 2.0])
    p_LoLq_L = np.array([0.0, 0.0, load_radius])          # top of load sphere

    rope = builder.AddSystem(
        TensionOnlyRope(
            plant=plant,
            quad_body=quad,
            load_body=load,
            p_QoQq_Q=p_QoQq_Q,
            p_LoLq_L=p_LoLq_L,
            L0=rope_L0,
            k=rope_k,
            c=rope_c,
        )
    )

    builder.Connect(plant.get_state_output_port(), rope.get_input_port(0))
    builder.Connect(rope.get_output_port(0), plant.get_applied_spatial_force_input_port())

    tension_logger = LogVectorOutput(rope.get_output_port(1), builder)

    # Feedforward: slightly above quad weight.
    u_ff_z = quad_mass * g + 10.0

    ctrl = builder.AddSystem(
        QuadXZController(
            plant=plant,
            x_joint=x_joint,
            z_joint=z_joint,
            x0=x_quad0,
            x_left=x_left,
            x_right=x_right,
            z0=z_quad0,
            z_final=z_quad_final,
            ramp_start=ramp_start,
            climb_rate=climb_rate,
            move_pause=move_pause,
            move_rate=move_rate,
            kp=200.0,
            kd=60.0,
            u_max_x=200.0,      # N
            u_max_z=400.0,      # N
            u_ff_z=u_ff_z,
        )
    )
    builder.Connect(plant.get_state_output_port(), ctrl.get_input_port(0))
    builder.Connect(ctrl.get_output_port(0), plant.get_actuation_input_port())

    meshcat = StartMeshcat()
    # When re-running the script, Meshcat can keep old geometry/lines around.
    # Clearing here avoids a stale rope color/pose from the previous run.
    meshcat.Delete()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Rope visualization (a line segment between attachment points).
    rope_viz = builder.AddSystem(
        RopeLineVisualizer(
            plant=plant,
            quad_body=quad,
            load_body=load,
            p_QoQq_Q=p_QoQq_Q,
            p_LoLq_L=p_LoLq_L,
            meshcat=meshcat,
            path="rope",
            L0=rope_L0,
        )
    )
    builder.Connect(plant.get_state_output_port(), rope_viz.get_input_port(0))

    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Update hover feedforward to include the visual model's inertial mass.
    quad_visual_mass = plant.CalcTotalMass(plant_context, [quad_visual_instance])
    ctrl.set_u_ff_z((quad_mass + quad_visual_mass) * g + 10.0)

    x_joint.set_translation(plant_context, x_quad0)
    x_joint.set_translation_rate(plant_context, 0.0)
    z_joint.set_translation(plant_context, z_quad0)
    z_joint.set_translation_rate(plant_context, 0.0)

    X_WL0 = RigidTransform([0.0, 0.0, load_radius])
    plant.SetFreeBodyPose(plant_context, load, X_WL0)
    plant.SetFreeBodySpatialVelocity(
        plant_context,
        load,
        SpatialVelocity(w=np.zeros(3), v=np.zeros(3)),
    )

    # Ensure visuals (including the rope line) reflect the initial state right away.
    diagram.ForcedPublish(context)

    simulator.set_target_realtime_rate(1.0)
    if enable_recording:
        meshcat.StartRecording()
    simulator.AdvanceTo(sim_duration)
    if enable_recording:
        meshcat.StopRecording()
        meshcat.PublishRecording()

    t = tension_logger.FindLog(context).sample_times()
    y = tension_logger.FindLog(context).data()[0, :]
    print(f"Logged {len(t)} tension samples.")
    print(f"Max tension: {y.max():.2f} N (should be ~0 at start, then increase when taut).")


if __name__ == "__main__":
    main()
