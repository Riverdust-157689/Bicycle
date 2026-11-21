"""Build legacy assisted B or the contact-only full-body variant.

Legacy scenes retain their existing constraints. The full-body branch has no
equalities, a raised seat, and an independently switchable crank torque motor.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from bicycle_sim.cli.check_rider_steering import build_rig
from bicycle_sim.control.hand_synergy import FingerSynergy, remove_palm_connections
from bicycle_sim.models.bicycle import numbers, sub
from bicycle_sim.models.tire import set_ellipsoid_tire_geoms, set_wheel_ground_rolling_contact

ARM_PARTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
LEG_PARTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
WAIST_PARTS = ("yaw", "roll", "pitch")


def model_fingerprint(m):
    """Pose-independent fingerprint for the free-bike initialization model."""
    digest = hashlib.sha256(m.names)
    for name in (
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "geom_type",
        "geom_dataid",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "geom_condim",
        "geom_priority",
        "geom_margin",
        "geom_gap",
        "geom_solref",
        "geom_solimp",
        "mesh_vert",
        "jnt_type",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "dof_damping",
        "dof_frictionloss",
        "dof_armature",
        "body_ipos",
        "body_iquat",
        "exclude_signature",
        "actuator_gear",
        "actuator_trnid",
        "actuator_gaintype",
        "actuator_biastype",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
    ):
        digest.update(np.asarray(getattr(m, name)).tobytes())
    digest.update(
        np.asarray(
            [
                m.opt.timestep,
                m.opt.iterations,
                m.opt.integrator,
                m.opt.solver,
                m.opt.cone,
                *m.opt.gravity,
            ]
        ).tobytes()
    )
    # Preserve historical hashes for pair-free models, but never accept an
    # initialization from a different explicit wheel/ground contact model.
    if m.npair:
        for name in (
            "pair_dim",
            "pair_geom1",
            "pair_geom2",
            "pair_friction",
            "pair_solref",
            "pair_solreffriction",
            "pair_solimp",
            "pair_margin",
            "pair_gap",
        ):
            digest.update(np.asarray(getattr(m, name)).tobytes())
    return digest.hexdigest()


def build_rider_model(
    cfg, reference_path=Path("output/phase1/rider_kinematic_reference.npz"), *, load_settled=True
):
    if cfg["experiment"] != "B":
        raise ValueError("Expected experiment B configuration")
    tire_collision = cfg.get("tire_collision_model", "mesh")
    if tire_collision not in ("mesh", "ellipsoid"):
        raise ValueError("tire_collision_model must be mesh or ellipsoid")
    m, q0, ctrl0 = build_rig(fixed_frame=False)
    # mj_saveLastXML preserves the compiled model and resolved mesh paths.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="bicycle-b-", dir="/tmp") as tmp:
        xml = Path(tmp) / "scene.xml"
        mujoco.mj_saveLastXML(str(xml), m)
        root = ET.parse(xml).getroot()
    keys = root.find("keyframe")
    if keys is not None:
        root.remove(keys)
    natural_grasp = cfg.get("hand_control_enabled", False)
    whole_body = cfg.get("whole_body_control", False)
    if whole_body:
        if not natural_grasp:
            raise ValueError("Whole-body task requires contact hands")
        equality = root.find("equality")
        seat_weld = equality.find("weld[@name='diagnostic_seat_connection']")
        equality.remove(seat_weld)
        rise = cfg["saddle_raise_m"]
        seat = root.find(".//geom[@name='saddle']")
        p = m.geom_pos[m.geom("saddle").id].copy()
        p[2] += rise
        seat.set("pos", numbers(p))
        post = root.find(".//geom[@name='seat_post']")
        p = m.geom_pos[m.geom("seat_post").id].copy()
        size = m.geom_size[m.geom("seat_post").id].copy()
        p[2] += rise / 2
        size[1] += rise / 2
        post.attrib.pop("fromto", None)
        post.set("pos", numbers(p))
        post.set("size", numbers(size[:2]))
        site = root.find(".//site[@name='saddle_contact']")
        p = m.site_pos[m.site("saddle_contact").id].copy()
        p[2] += rise
        site.set("pos", numbers(p))
    if natural_grasp:
        if (
            not 0 <= cfg["hand_nominal_closure"] <= 1
            or not 0 < cfg["hand_closure_action_scale"] <= 1
        ):
            raise ValueError("Invalid nominal hand closure/action scale")
        if cfg["finger_force_scale_N"] <= 0 or cfg["finger_contact_reward_weight"] < 0:
            raise ValueError("Invalid contact feature/reward scaling")
        remove_palm_connections(root)
    if tire_collision == "ellipsoid":
        set_ellipsoid_tire_geoms(root, m)
    set_wheel_ground_rolling_contact(root, m, cfg.get("wheel_ground_rolling_friction_m", 0.0))
    # The locomotion asset has four corner spheres only. A pedal centered at
    # the sole reference can sit between all four, so add the missing sole
    # surface in this task adapter. Explicit body inertias stay unchanged.
    for side in ("left", "right"):
        foot = root.find(f".//body[@name='{side}_ankle_roll_link']")
        sub(
            foot,
            "geom",
            name=f"{side}_riding_sole",
            type="box",
            pos="0.035 0 -0.025",
            size="0.09 0.035 0.01",
            density="0",
            friction="0.6 0 0",
            condim="3",
            priority="1",
            rgba="0.2 0.2 0.2 0",
            group="3",
        )
    drive = sub(
        root.find("actuator"),
        "velocity",
        name="crank_drive",
        joint="crank_spin",
        kv=cfg["crank_servo_kv"],
        forcelimited="true",
        forcerange=numbers([-cfg["max_crank_torque_Nm"], cfg["max_crank_torque_Nm"]]),
    )
    if whole_body:
        # A torque motor can be exactly off. Tracking current velocity with a
        # velocity actuator would retain implicit damping in the integrator.
        drive.tag = "motor"
        del drive.attrib["kv"]
    # The grip/seat constraints must be defined relative to the mounted pose,
    # never relative to an accidental reset pose or a world body.
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    ctrl0 = np.r_[ctrl0, 0 if whole_body else cfg["crank_speed_rad_s"]]
    arms = [m.joint(f"{s}_{p}_joint").id for s in ("left", "right") for p in ARM_PARTS]
    legs = [m.joint(f"{s}_{p}_joint").id for s in ("left", "right") for p in LEG_PARTS]
    robot_act = [
        a
        for a in range(m.nu)
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
        and m.actuator(a).name != "crank_drive"
    ]
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Generate the phase reference first: uv run --locked python -I -m bicycle_sim.control.rider_fit"
        )
    with np.load(reference_path) as ref:
        phases, poses = ref["phases"], ref["qpos"]
    if poses.shape != (25, m.nq) or not np.allclose(phases, np.linspace(0, 2 * np.pi, 25)):
        raise ValueError(
            "Phase reference does not match the mounted rider model; regenerate rider_fit"
        )
    if not np.isfinite(poses).all() or not np.allclose(
        poses[0, m.jnt_qposadr[legs]], q0[m.jnt_qposadr[legs]], atol=1e-6
    ):
        raise ValueError("Invalid/stale phase reference; regenerate rider_fit")
    maps = {
        "arm_q": m.jnt_qposadr[arms],
        "arm_v": m.jnt_dofadr[arms],
        "arm_act": np.array([m.actuator(m.joint(j).name).id for j in arms]),
        "leg_act": np.array([m.actuator(m.joint(j).name).id for j in legs]),
        "robot_q": m.jnt_qposadr[m.actuator_trnid[robot_act, 0]],
        "robot_v": m.jnt_dofadr[m.actuator_trnid[robot_act, 0]],
        "leg_table": poses[:, m.jnt_qposadr[legs]],
    }
    if natural_grasp:
        hand = FingerSynergy(
            m, q0, kp=cfg["hand_kp"], kv=cfg["hand_kv"], closure_rate=cfg["hand_closure_rate_per_s"]
        )
        ref = json.loads(Path(cfg["grasp_reference_path"]).read_text())
        if not np.isclose(ref["nominal_finger_closure"], cfg["hand_nominal_closure"]):
            raise ValueError("Grasp reference closure differs from task nominal")
        for name, values in ref["joint_qpos"].items():
            joint = m.joint(name)
            size = 7 if joint.type[0] == mujoco.mjtJoint.mjJNT_FREE else 1
            value = np.asarray(values, dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid grasp pose: {name}")
            q0[joint.qposadr[0] : joint.qposadr[0] + size] = value
        for name, value in ref["control_targets"].items():
            ctrl0[m.actuator(name).id] = value
        np.testing.assert_allclose(
            ctrl0[hand.aids], hand.targets(np.full(6, cfg["hand_nominal_closure"])), atol=1e-6
        )
        maps.update(
            hand_act=hand.aids,
            hand_q=hand.qids,
            hand_v=hand.vids,
            hand_open=hand.nominal,
            hand_closed=hand.closed,
            hand_columns=hand.columns,
        )
    if whole_body:
        waist = [m.joint(f"waist_{p}_joint").id for p in WAIST_PARTS]
        maps.update(
            leg_q=m.jnt_qposadr[legs],
            leg_v=m.jnt_dofadr[legs],
            pedal_q=np.array([m.joint(f"{s}_pedal_spin").qposadr[0] for s in ("left", "right")]),
            pedal_v=np.array([m.joint(f"{s}_pedal_spin").dofadr[0] for s in ("left", "right")]),
            waist_q=m.jnt_qposadr[waist],
            waist_v=m.jnt_dofadr[waist],
            waist_act=np.array([m.actuator(m.joint(j).name).id for j in waist]),
        )
        if m.neq:
            raise ValueError("Whole-body scene must contain no equality constraints")
        if load_settled:
            path = Path(cfg["whole_body_reference_path"])
            if not path.is_file():
                raise FileNotFoundError(f"Generate the GPU no-weld initialization first: {path}")
            settled = json.loads(path.read_text())
            if settled["model_fingerprint"] != model_fingerprint(m):
                raise ValueError("No-weld reference does not match the current physical model")
            if settled["backend"] != "mjlab/MuJoCo-Warp" or not settled["passed"]:
                raise ValueError("Training requires an accepted GPU no-weld reference")
            if set(settled["joint_qpos"]) != {m.joint(j).name for j in range(m.njnt)}:
                raise ValueError("Settled reference must contain every joint")
            if set(settled["control_targets"]) != {m.actuator(a).name for a in range(m.nu)}:
                raise ValueError("Settled reference must contain every control target")
            if (
                settled["phase_reference_sha256"]
                != hashlib.sha256(reference_path.read_bytes()).hexdigest()
            ):
                raise ValueError("No-weld reference phase table changed")
            leg_reference = np.asarray(settled.get("leg_reference"), dtype=float)
            if (
                leg_reference.shape != (25, 12)
                or not np.isfinite(leg_reference).all()
                or not np.allclose(leg_reference[0], leg_reference[-1], atol=1e-6)
            ):
                raise ValueError("Invalid fitted seated leg reference")
            if np.any(leg_reference < m.jnt_range[legs, 0] - 1e-6) or np.any(
                leg_reference > m.jnt_range[legs, 1] + 1e-6
            ):
                raise ValueError("Seated leg reference violates joint limits")
            maps["leg_table"] = leg_reference
            for name, values in settled["joint_qpos"].items():
                j = m.joint(name)
                n = 7 if j.type[0] == mujoco.mjtJoint.mjJNT_FREE else 1
                value = np.asarray(values)
                if value.shape != (n,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid settled pose: {name}")
                q0[j.qposadr[0] : j.qposadr[0] + n] = value
            for name, value in settled["control_targets"].items():
                if not np.isfinite(value):
                    raise ValueError("Nonfinite settled control")
                ctrl0[m.actuator(name).id] = value
            ctrl0[m.actuator("crank_drive").id] = 0
    return m, q0, ctrl0, maps
