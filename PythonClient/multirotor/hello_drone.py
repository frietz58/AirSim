import setup_path 
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
import random

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

imu_data = client.getImuData()
s = pprint.pformat(imu_data)
print("imu_data: %s" % s)

barometer_data = client.getBarometerData()
s = pprint.pformat(barometer_data)
print("barometer_data: %s" % s)

magnetometer_data = client.getMagnetometerData()
s = pprint.pformat(magnetometer_data)
print("magnetometer_data: %s" % s)

gps_data = client.getGpsData()
s = pprint.pformat(gps_data)
print("gps_data: %s" % s)

print()
target_pose = client.simGetObjectPose("SimpleTarget")
print("Target location: {}".format(target_pose))

new_target_position = target_pose.position
new_target_position.x_val = random.randint(-6, 6)
new_target_position.y_val = random.randint(-6, 6)
new_target_position.z_val = random.randint(3, 4)

# there is a bug here, z value appears to be inverted?
new_target_position.z_val = new_target_position.z_val * -1

target_pose.position = new_target_position
client.simSetObjectPose("SimpleTarget", target_pose)
print("New target location: {}".format(target_pose))
print()

# client.simSetObjectPose("SimpleTarget", )

airsim.wait_key('Press any key to takeoff')
client.takeoffAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to move vehicle to (5, 0, 0) at 5 m/s')
client.moveToPositionAsync(5, 0, 0, 5).join()



client.hoverAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to take images')
# get camera images from the car
responses = client.simGetImages([
    # https://github.com/microsoft/AirSim/issues/3553
    # airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
    # airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
    # airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
    # airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
    airsim.ImageRequest("0", 3),  # depth visualization image
    airsim.ImageRequest("1", 2),  # depth in perspective projection
    airsim.ImageRequest("1", 0),  # scene vision image in png format
    airsim.ImageRequest("1", 0, False, False)])  # scene vision image in uncompressed RGBA array
print('Retrieved images: %d' % len(responses))

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

for idx, response in enumerate(responses):

    filename = os.path.join(tmp_dir, str(idx))

    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
    elif response.compress: #png format
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    else: #uncompressed array
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
        cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

airsim.wait_key('Press any key to reset to original state')

client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
