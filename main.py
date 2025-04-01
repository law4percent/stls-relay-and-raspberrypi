from stls_lib import stls

def handle_invalid_input(input_name, expected_values, value):
    print(f"\nInvalid input found at {input_name}. Input must be one of {expected_values}. Found: {value}")
    exit()

def process_rp_device(data):
    """
    Process the Raspberry Pi device logic.
    """
    from stls_lib.rp import rp_write_points, rp_process_video
    
    write_points_mode = data["write_points_mode"].lower()
    if write_points_mode == "true":
        rp_write_points.main(
                save_path = data["zones_file_path"],
                frame_height = data["frame_height"],
                frame_width = data["frame_width"],
                ord_key = data["ord_key"],
                max_zones = data["max_zones"]
            )
        exit()  # Exit after completing write_points

    elif write_points_mode == "false":
        rp_process_video.main(
                weight_file_path = data["weight_file_path"],
                class_list_file_path = data["class_list_file_path"],
                zones_file_path = data["zones_file_path"],
                detect_sensitivity = data["detect_sensitivity"],
                frame_name = data["frame_name"],
                time_interval = data["time_interval"],
                frame_height = data["frame_height"],
                frame_width = data["frame_width"],
                wait_key = data["wait_key"],
                ord_key = data["ord_key"]
            )
    else:
        handle_invalid_input("data[\"write_points_mode\"]", ["true", "false"], write_points_mode)


def main():
    """
    Main entry point for the script.
    """
    data = stls.extract_root_data(file_path="src/utils/root_data.txt")
    process_rp_device(data)

if __name__ == "__main__":
    main()
