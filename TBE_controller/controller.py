# Importing libraries
import time
import numpy as np
from typing import Optional
from scipy.interpolate import PchipInterpolator

from utilities import *
from data_obtainer import SensorData


# Creating a controller class
class TBEController:

    def __init__(self, logger: Logger):

        # median stride time
        self.stride_time = float('nan')
        # self.stride_time = 1.0

        # time params
        self._init_time = time.perf_counter()   

        # Torque profile
        self.torque_profile: Optional[PchipInterpolator] = None

        # Logger for logging controller data
        self.logger = logger

        # Track last heel strike time for phase calculation in TBEActivation
        self.last_heel_strike_time = float('nan')

        self.logger.logger.info("TBE Controller initialized.") 
    
# Creating the calibration class for calibrating the thresholds for heel strike and toe off detection
class TBECalibration():

    def __init__(self, controller: TBEController, data: SensorData, logger: Logger):
        self.controller = controller
        self.data = data
        self.logger = logger

        # heel strike and toe off time points
        self.heel_strike_times = np.zeros(NUM_STRIDES)   
        self.toe_off_times = np.zeros(NUM_STRIDES)       

        # Threshold for detecting heel strike and toe off
        self.heel_strike_threshold = HEEL_STRIKE_THRESHOLD
        self.toe_off_threshold = TOE_OFF_THRESHOLD

        # To store previous states
        self._prev_heel_state = False
        self._prev_toe_state = False

        # To initialize edge rise and fall detection
        self._edge_initialized = False

        # Torque profile parameters
        self.current_torque_profile_points = []
        self.torque_profile_points = []
        self.stride_saved = False

        # TO check whether calibration is done or not
        self.calibrated = False
        
        # Checking phase of calibration
        self._stride_times_collected = False

        # start time of current stride for torque profile collection
        self._torque_stride_start_time = None

    # Detect heel strike from heel FSR rising above threshold
    def detectHeelGround(self) -> bool:
        return self.data.filtered_heel_fsr > self.heel_strike_threshold

    # Detect toe off from toe FSR falling below threshold
    def detectToeOff(self) -> bool:
        return self.data.filtered_toe_fsr < self.toe_off_threshold

    # Checks rising threshold of heel strike
    def detectHeelStrikeEdge(self) -> bool:
        current = self.detectHeelGround()
        rising_edge = current and not self._prev_heel_state
        self._prev_heel_state = current
        return rising_edge

    # Checks falling threshold of toe off
    def detectToeOffEdge(self) -> bool:
        current = self.detectToeOff()
        falling_edge = current and not self._prev_toe_state
        self._prev_toe_state = current
        return falling_edge

    # Detect the phase of the gait cycle based on heel strike and toe off detections
    def detectStancePhase(self) -> bool:
        if self.detectHeelGround() or not self.detectToeOff():
            return True
        
        return False
    
    # Calculate the median stride time from the stride time array
    def getStanceTime(self) -> None:

        # Prime edge states on first call to avoid duplicates
        if not self._edge_initialized:
            self._prev_heel_state = self.detectHeelGround()
            self._prev_toe_state = self.detectToeOff()
            self._edge_initialized = True
            return

        # Detect transitions
        heel_strike_edge = self.detectHeelStrikeEdge()
        toe_off_edge = self.detectToeOffEdge()

        # Record heel strike timestamp on rising edge, along with no noise check
        if heel_strike_edge:

            heel_strike_time = self.logger.getCurrentTime()
            if (heel_strike_time - self.heel_strike_times[-1] > 0.5):
                self.logger.logger.info("Heel strike detected. Entering stance phase.")
                self.heel_strike_times = np.roll(self.heel_strike_times, -1)
                self.heel_strike_times[-1] = heel_strike_time
        
        # Record toe off timestamp on falling edge, along with no noise check
        if toe_off_edge and self.heel_strike_times[-1] > 0:
            self.logger.logger.info("Toe off detected. Entering swing phase.")
            self.toe_off_times = np.roll(self.toe_off_times, -1)
            self.toe_off_times[-1] = self.logger.getCurrentTime()
       
        # Once we have NUM_STRIDES worth of timestamps, compute stride time
        if (self.heel_strike_times[0] > 0) and (self.toe_off_times[0] > 0):
            self.controller.stride_time = float(np.median(self.toe_off_times - self.heel_strike_times))
            self._stride_times_collected = True    
            self.logger.logger.info(f"Median stride time calculated: {self.controller.stride_time:.2f} seconds.")                                                

    
    # Calculate the torque profile based on the stance phase percent and the stride time
    def calculateTorqueProfile(self) -> None:

        # 1. Simulation generated torque profile data

        # common_grid = np.linspace(0, 1, 101)  
        # data = scipy.io.loadmat('avg_torque_profiles.mat')

        # avg_profiles = data['avg_profiles']  # shape: (101, 2)

        # self.controller.torque_profile = PchipInterpolator(common_grid, avg_profiles[:, 0]) 

        # 2. Real collected torque profile data
        
        # data = scipy.io.loadmat('ankle_data_clipped.mat')
        # ankle_r = data['ankle_r'][0]  # shape: (200, )

        # common_grid = np.linspace(0, 1, len(ankle_r))

        # self.controller.torque_profile = PchipInterpolator(common_grid, ankle_r)

        # 3. From real data, taking 6 points, and splining the data
        phases  = TAU_PHASE_ARRAY
        torques = TAU_VAL_ARRAY * PEAK_TORQUE 

        self.controller.torque_profile = PchipInterpolator(phases, torques)


    # The actual calibration process
    def calibrate(self) -> None:

        # Phase 1: Collect stride times
        if not self._stride_times_collected:
            self.getStanceTime()
            return                                                
        
        # Phase 2: Build the torque profile interpolator
        if self.controller.torque_profile is None:
            self.calculateTorqueProfile()
            self.calibrated = True           


# Class for activating TBE controller
class TBEActivation():

    def __init__(self, controller: TBEController, data: SensorData, calibration: TBECalibration, logger: Logger):
        self.controller = controller
        self.data = data
        self.calibration = calibration
        self.logger = logger

        # TO store the recent strides for adaptive stride time calculation
        self._recent_strides = []

    # Function to provide torque output based on the current time and the torque profile
    def giveTorqueOutput(self, phase: float) -> None:
        if self.controller.torque_profile is not None:
            self.data.torque_input += float(self.controller.torque_profile(phase))
         
            self.data.sendTorqueData()

    # Function to activate the controller based on the current phase of the gait cycle
    def activate(self) -> None:       

        # Keep tracking heel strikes to update phase reference
        if self.calibration.detectHeelStrikeEdge():
            self.controller.last_heel_strike_time = self.logger.getCurrentTime()
            self.logger.logger.info("Heel strike detected (activation). Phase reset.")


        # Check for heel strike reference and stride time to compute phase
        if np.isnan(self.controller.last_heel_strike_time):
            return
        if np.isnan(self.controller.stride_time):
            return

        elapsed = self.logger.getCurrentTime() - self.controller.last_heel_strike_time
        phase = elapsed / self.controller.stride_time
        phase = float(np.clip(phase, 0.0, 1.0))        

        # Command torque during stance, and not in swing
        if not self.calibration.detectStancePhase():
            self.data.torque_input = 0.0
            #  When emergency stop is pressed
            if self.data.can_bus is None:
                self.logger.logger.info("Emergency stop activated. Stopping torque output.")
                feedback = self.data.read_feedback()
                if feedback is not None:
                    self.data._torque_plot = -self.data.torque_input
                else:
                    self.data._torque_plot = 0.0  
                return
            else:
                self.data.sendTorqueData()
                return
        
        if phase < 1.0:
            self.giveTorqueOutput(phase)  

# Class for the safety impedance controller to keep ankle joint angle within safe limits (not implemented in current version, but can be extended in future)
class TBEImpedanceController():

    def __init__(self, controller: TBEController, data: SensorData, logger: Logger):
        self.controller = controller
        self.data = data
        self.logger = logger

    def checkLimits(self) -> None:
        
        # Encoder limit val
        hyper_flexion_value = self.data.encoder_data - np.clip(self.data.encoder_data, DORSIFLEXION_LIMIT, PLANTARFLEXION_LIMIT)
        # Check if encoder data is within limits and if not, command opposing impedance torque
        if not (hyper_flexion_value == 0):
            self.logger.logger.warning("Joint limit exceeded! Applying safety impedance control.")
            # Simple PD impedance control
            torque_command = KP_IMPEDANCE * hyper_flexion_value + KD_IMPEDANCE * self.data.encoder_velocity
            # self.data.sendTorqueData(torque_command)
            self.data.torque_input += torque_command  
