from picamera2 import Picamera2
import cv2
import numpy as np

points = []
entry_counter = 0
frame_h = 0
frame_w = 0

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw_frame(points, frame_copy, frame_name)

def redraw_frame(points, image, frame_name):
    temp_image = image.copy()
    if len(points) > 1:
        cv2.polylines(temp_image, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        instruction(temp_image)
        
    for point in points:
        cv2.circle(temp_image, point, 5, (0, 0, 255), -1)
        instruction(temp_image)

    instruction(temp_image)
    cv2.imshow(frame_name, temp_image)

def save_points_to_file(file_path, max_zones):
    global points, entry_counter, frame_h, frame_w
    if entry_counter >= max_zones:
        print(f"Maximum number of zones ({max_zones}) reached. Program will exit.")
        return False
        
    with open(file_path, "a") as file:
        if entry_counter == 0:  # Only write the header when the first zone is saved
            file.write("zones: \n")
        
        formatted_points = ', '.join([f"({x}, {y})" for x, y in points])
        file.write(f"   {entry_counter}: [{formatted_points}]\n")
        
        if entry_counter == max_zones - 1:  # When the last zone is saved, write the footer
            file.write(f"\nnumber_of_zone: {entry_counter + 1}\n")
            file.write(f"frame_width: {frame_w}\n")
            file.write(f"frame_height: {frame_h}\n")
    
    print(f"Entry {entry_counter} saved to '{file_path}'.")
    points = []
    entry_counter += 1
    return True

def instruction(frame):
    cv2.rectangle(frame, (20, 10), (730, 65), (255, 255, 255), -1)
    cv2.putText(frame, f"Left click to select points.", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(frame, f"Press 's' to save, 'c' to close the polygon, 'u' to undo last point, and 'q' to quit.", (25, 30 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

def main(save_path, frame_height, frame_width, ord_key, max_zones):
    global points, frame_copy, frame_name, frame_h, frame_w

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (frame_width, frame_height)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    frame_name = "Writing Points Mode: RP"
    frame_idx = 0

    frame_h = frame_height
    frame_w = frame_width

    print("Press 'n' to move to the next frame, 's' to save points, 'c' to close the polygon, and 'q' to quit.")

    while True:
        frame = picam2.capture_array()
        frame_copy = frame.copy()

        instruction(frame)

        cv2.imshow(frame_name, frame)
        cv2.setMouseCallback(frame_name, click_event)
        
        print(f"Frame {frame_idx}: Left click to select points. Press 's' to save, 'c' to close the polygon, 'u' to undo last point, and 'q' to quit.")
        print(f"Zones created: {entry_counter}/{max_zones}")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Save points to file
                if not save_points_to_file(save_path, max_zones):
                    # If max zones reached, cleanup and exit
                    picam2.stop()
                    cv2.destroyAllWindows()
                    return
                redraw_frame(points, frame_copy, frame_name)
            elif key == ord('c'):  # Close the polygon
                if len(points) > 2:
                    cv2.polylines(frame_copy, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)
                    redraw_frame(points, frame_copy, frame_name)
                    print("Polygon closed.")
                else:
                    print("A polygon must have at least 3 points.")
            elif key == ord('n'):  # Next frame
                points = []
                frame_idx += 1
                break
            elif key == ord('u'):  # Undo last point
                if points:
                    points.pop()  # Remove the last point
                    print(f"Undo last point. Remaining points: {len(points)}")
                    redraw_frame(points, frame_copy, frame_name)
                else:
                    print("No points to undo.")
            elif key == ord(ord_key):  # Quit
                picam2.stop()
                cv2.destroyAllWindows()
                return