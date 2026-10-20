import sys
from pathlib import Path

from controller import TBEController, TBECalibration, TBEActivation, TBEImpedanceController
from data_obtainer import SensorData
from utilities import *

# dashboard/ is a sibling of TBE_controller/ in the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard.backend.run_logger import RunLogger, ModelInfo


def main():

    # Start logging
    TBElog = Logger()

    # Start run tracking: every launch of main.py is one "run", recorded to
    # Parquet + SQLite, and relayed live to the dashboard if it's running.
    #
    # The current TBE controller uses a fixed torque profile, not a learned
    # model, so there's no W&B training run behind it yet. We still record
    # its parameters as `architecture` so the dashboard can diff trials
    # (e.g. "assistance_level changed from 0.1 to 0.15 between these runs").
    # When a learned controller (e.g. a trained policy) replaces this, fill
    # in wandb_run_url/wandb_run_id/checkpoint_ref so trials link straight
    # to that model's training history.
    run_logger = RunLogger(
        meta={
            "assistance_level": ASSISTANCE_LEVEL,
            "peak_torque": PEAK_TORQUE,
            "motor_control_freq": MOTOR_CONTROL_FREQ,
        },
        model_info=ModelInfo(
            name="tbe-fixed-torque-profile",
            wandb_run_url=None,   # set this once a learned model/policy is in the loop
            wandb_run_id=None,
            wandb_project=None,
            architecture={
                "tau_phase_array": list(TAU_PHASE_ARRAY),
                "tau_val_array": list(TAU_VAL_ARRAY),
                "assistance_level": ASSISTANCE_LEVEL,
                "peak_torque": PEAK_TORQUE,
                "kp_impedance": KP_IMPEDANCE,
                "kd_impedance": KD_IMPEDANCE,
            },
            checkpoint_ref=None,
        ),
    )
    run_logger.start()
    TBElog.logger.info(f"Run ID: {run_logger.run_id}")

    TBElog.logger.info("Opening sensor channels...")
    # Create an instance of the SensorData class to read the sensor data
    sensor_data = SensorData(TBElog)

    TBElog.logger.info("Instantiating TBE Controller...")
    # Create an instance of the TBEController
    controller = TBEController(TBElog)

    TBElog.logger.info("Instantiating TBE Calibration...")
    # Create an instance of the TBECalibration class to calibrate the thresholds for heel strike and toe off detection
    calibration = TBECalibration(controller, sensor_data, TBElog)

    TBElog.logger.info("Instantiating TBE Activation...")
    # Create an instance of the TBEActivation class to activate the controller
    activation = TBEActivation(controller, sensor_data, calibration, TBElog)

    # Create an instance of the TBEImpedanceController class to run impedance control
    impedance_controller = TBEImpedanceController(controller, sensor_data, TBElog)
    TBElog.logger.info("Instantiating TBE Impedance Controller...")

    TBElog.logger.info(f"Starting main loop at {MOTOR_CONTROL_FREQ} Hz...")

    # Start the main loop
    try:
        while True:

            loop_start = time.perf_counter()
            
            # Read the sensor data
            sensor_data.readSensors()

            # If not calibrated, run calibration
            if not calibration.calibrated:
                calibration.calibrate()
            else:

                # Run impedance control
                # impedance_controller.checkLimits()
                # If calibrated, run activation
                activation.activate()

            run_logger.log_sample(
                t=loop_start,
                heel_fsr=sensor_data.filtered_heel_fsr,
                toe_fsr=sensor_data.filtered_toe_fsr,
                ankle_angle=sensor_data.encoder_data,
                ankle_velocity=sensor_data.filtered_encoder_velocity,
                torque_cmd=sensor_data._torque_plot,
                calibrated=calibration.calibrated,
                stride_time=controller.stride_time,
            )

            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed
            # Print the values for elapsed and time left
            # TBElog.logger.info(f"THe time elapsed: {elapsed}, sleep time : {sleep_time}")
            # If the loop is running faster than the desired frequency, sleep for the remaining time
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        TBElog.logger.info("Shutting down Controller...")
        sensor_data.shutdown()
        run_logger.stop(status="completed")
        TBElog.logger.info("Exiting.")

    except Exception:
        run_logger.stop(status="crashed")
        raise

     
if __name__ == "__main__":
    main()