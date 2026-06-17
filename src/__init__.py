"""AGV Rail Defect Detection -- Inspection & Smart Maintenance System (PoC).

Software-only proof-of-concept package. See README.md for the architecture and
the detection contract. Subpackages / modules:

    schema         shared Pydantic message models (the wire contract)
    pipeline       end-to-end orchestration (detect -> prescribe -> localise -> persist -> publish)
    sources        swappable frame sources       (video/images -> camera)
    inference      swappable detectors           (Ultralytics YOLO -> TFLite/Coral)
    localisation   swappable localisation        (sim GPS CSV -> GPS/odometry/IMU)
    prescriptive   rule-based severity/urgency/action engine (the project's novelty)
    messaging      MQTT publishing / subscribing
    storage        SQLite persistence
    dashboard      Flask + Socket.IO operator dashboard
"""

__version__ = "0.1.0"
