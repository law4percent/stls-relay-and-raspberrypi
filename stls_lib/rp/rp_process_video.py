import cv2
import time
from picamera2 import Picamera2
from stls_lib import stls
import RPi.GPIO as GPIO

# Define relay pin numbers
RELAY_ZONE_0_CAR = 17
RELAY_ZONE_0_MOTORBIKE = 27
RELAY_ZONE_0_OTHER = 22

# Setup GPIO mode and initialize relays
# Setup GPIO mode and initialize relays
GPIO.setmode(GPIO.BCM)
GPIO.setup([RELAY_ZONE_0_CAR, RELAY_ZONE_0_MOTORBIKE, RELAY_ZONE_0_OTHER], GPIO.OUT, initial=GPIO.HIGH)

    # Default: Turn all relays OFF
GPIO.output(RELAY_ZONE_0_CAR, GPIO.HIGH)
GPIO.output(RELAY_ZONE_0_MOTORBIKE, GPIO.HIGH)
GPIO.output(RELAY_ZONE_0_OTHER, GPIO.LOW)

def activate_relay(vehicle_type):
    print(vehicle_type)
    
    if vehicle_type == "car":
        GPIO.output(RELAY_ZONE_0_CAR, GPIO.LOW)
        GPIO.output(RELAY_ZONE_0_MOTORBIKE, GPIO.HIGH)
        GPIO.output(RELAY_ZONE_0_OTHER, GPIO.HIGH)
    elif vehicle_type == "motorbike":
        GPIO.output(RELAY_ZONE_0_CAR, GPIO.HIGH)
        GPIO.output(RELAY_ZONE_0_MOTORBIKE, GPIO.LOW)
        GPIO.output(RELAY_ZONE_0_OTHER, GPIO.HIGH)
    elif vehicle_type == "none":
        GPIO.output(RELAY_ZONE_0_CAR, GPIO.HIGH)
        GPIO.output(RELAY_ZONE_0_MOTORBIKE, GPIO.HIGH)
        GPIO.output(RELAY_ZONE_0_OTHER, GPIO.LOW)

def main(weight_file_path: str,
         class_list_file_path: str,
         zones_file_path: str,
         detect_sensitivity: float,
         time_interval: float,
         frame_name: str,
         frame_height: int,
         frame_width: int,
         wait_key: int,
         ord_key: str,
         ):

    # Initialize camera
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (frame_width, frame_height)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    
    # Load YOLO model and configurations
    yolo_model = stls.load_model(weight_file_path)
    class_list = stls.load_class_names(class_list_file_path)
    
        # Extract data from the zones.txt file
    data = stls.extract_data_from_file(zones_file_path)
    zones = stls.convert_coordinates(data["zones"], data["frame_width"], data["frame_height"], frame_width, frame_height) # Ensuring the zone coordinates to fit the new frame dimensions
    number_of_zones = data["number_of_zones"]

    # Initialize zones and tracking data
    count = 0
    success = True
    zones_data = {"countdown_start_time": 0.0, "refresh": False, "get_vehicle": 'none'}
    prev_vehic_zone = 'none'

    while success:
        start_time = time.time() * 1000
        curr_time = time.time()
        frame = picam2.capture_array()

        count += 1
        if count % 3 != 0:
            continue

        collected_vehicle = stls.init_list_of_collected_vehicle(number_of_zones)
        frame = cv2.resize(frame, (frame_width, frame_height))
        boxes = stls.get_prediction_boxes(frame, yolo_model, detect_sensitivity)
        stls.draw_polylines_zones(frame, zones, frame_name)  # Optional visualization
        collected_vehicle = stls.track_objects_in_zones(frame, boxes, class_list, zones, collected_vehicle, frame_name)

        hanlde_current_vehic = stls.handle_zone_queuing(collected_vehicle, curr_time, zones_data, time_interval)
        stls.traffic_light_display(frame, is_zone_occupied = len(collected_vehicle) > 0) # Optional visualization
            
        # Get the current vehicle types for each zone, or 'none' if invalid
        curr_vehic_zone = hanlde_current_vehic["vehicle"]

        if curr_vehic_zone != prev_vehic_zone:
            activate_relay(curr_vehic_zone)
            prev_vehic_zone = curr_vehic_zone

        data_to_display = {
            "frame_name": frame_name,
            "hanlde_current_vehic": hanlde_current_vehic,
            "processing_time": (time.time() * 1000) - start_time
        }
        stls.display_zone_info(frame, data_to_display)  # Optional visualization       
        success = stls.show_frame(frame, frame_name, wait_key, ord_key)  # Optional frame display

    cv2.destroyAllWindows()
    GPIO.cleanup()
