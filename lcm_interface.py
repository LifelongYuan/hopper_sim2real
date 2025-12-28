import lcm
from lcm_config.python.lrl_legged_data_lcmt import *
from lcm_config.python.lrl_lower_bodyimu_data_lcmt import *
from lcm_config.python.lrl_armwaisthead_data_lcmt import *
from lcm_config.python.lrl_hand_data_lcmt import *

from lcm_config.python.lrl_legged_cmd_lcmt import *
from lcm_config.python.lrl_armwaisthead_cmd_lcmt import *
from lcm_config.python.lrl_hand_cmd_lcmt import *
from lcm_config.python.gamepad_lcmt import *
import threading
from threading import Timer
import time
ON_BOARD = True
import signal
import sys
if not ON_BOARD:
    import rospy
    from sensor_msgs.msg import JointState
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry

def interrupt_handler(signum, frame):
    print("\nInterrupt received, shutting down...")
    sys.exit(0)
signal.signal(signal.SIGINT, interrupt_handler)
class robot_lcm:
    def __init__(self,
                publish_freq=520,
                ttl=1,
                pub_topic_name_dict={"leg":"controller2robot_legs",
                                    "ahw":"controller2robot_ahw",
                                    "hand":"controller2robot_hands"},
                sub_topic_name_dict={"sensor":"robot2controller_sensors",
                                    "leg":"robot2controller_legs",
                                    "ahw":"robot2controller_ahw",
                                    "hand":"robot2controller_hands"
                                },
                joint_state_ros_pub=True) -> None:
        self.pub_freq=publish_freq
        self._pub_name_dict:dict = pub_topic_name_dict
        self._sub_name_dict:dict = sub_topic_name_dict
        # filter dict by keys:
        self.filter()

        # clients
        self.r2c = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))
        self.c2r = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))

        # msgs
        self.c2r_leg_msg = lrl_legged_cmd_lcmt()
        self.c2r_ahw_msg = lrl_armwaisthead_cmd_lcmt()
        self.c2r_hand_msg = lrl_hand_cmd_lcmt()

        self.r2c_leg_msg = lrl_legged_data_lcmt()
        self.r2c_sensor_msg = lrl_lower_bodyimu_data_lcmt()
        self.r2c_ahw_msg = lrl_armwaisthead_data_lcmt()
        self.r2c_hand_msg = lrl_hand_data_lcmt()

        # TODO: two thread is enough?
        self.lcm_sender_thread = threading.Thread(target=self.thread_1)
        self.lcm_receiver_thread = threading.Thread(target=self.thread_2)
        self.s=time.time()

        if joint_state_ros_pub and not ON_BOARD:
 
            rospy.init_node("dog_joint_state_publisher", anonymous=True)
            self.joint_state_ros_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
            self.ros_joint_state_pub_thread = threading.Thread(target=self.thread_3)
            self.br = tf2_ros.TransformBroadcaster()
            self.ros_joint_state_pub_thread.start()

        # register handlers
        for key,t_name in self._sub_name_dict.items():
            self.r2c.subscribe(t_name, getattr(self,key+"_handler"))
        self._init_buffer()

    # handlers:
    def sensor_handler(self,c,d):
        self.r2c_sensor_msg = lrl_lower_bodyimu_data_lcmt.decode(d)
    def leg_handler(self,c,d):
        self.r2c_leg_msg = lrl_legged_data_lcmt.decode(d)
    def ahw_handler(self,c,d):
        self.r2c_ahw_msg = lrl_armwaisthead_data_lcmt.decode(d)
    def hand_handler(self,c,d):
        self.r2c_hand_msg = lrl_hand_data_lcmt.decode(d)

    def filter(self):
        """
        filter unused channels by keys
        """
        pass

    def publish(self):
        for key,t_name in self._pub_name_dict.items(): 
            self.c2r.publish(t_name, getattr(self,"c2r_"+key+"_msg").encode())
    
    def start_sending(self):
        self.lcm_sender_thread.start()

    def start_recving(self):
        self.lcm_receiver_thread.start()

    def thread_1(self):
        while 1:
            self.publish()
            elapsed_time = time.time() - self.s
            time.sleep(max(0, 1/self.pub_freq - elapsed_time))
            self.s = time.time()

    def thread_2(self):
        while 1:
            self.r2c.handle()

    def thread_3(self):
        # Initialize the node
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base"
    
        # Create a publisher for the JointState message
        # joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        # odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
    
        # Create a JointState message
        joint_state = JointState()
        # Set up the joint names - these should match your URDF
        joint_state.name = ["left_elbow_joint", "left_shoulder_yaw_joint",
            "left_shoulder_roll_joint", "left_shoulder_pitch_joint",
            "right_elbow_joint", "right_shoulder_yaw_joint",
            "right_shoulder_roll_joint", "right_shoulder_pitch_joint",
            "left_ankle_roll_joint", "left_ankle_pitch_joint", "left_knee_joint",
            "left_hip_pitch_joint", "left_hip_yaw_joint", "left_hip_roll_joint",
            "right_ankle_roll_joint", "right_ankle_pitch_joint", "right_knee_joint",
            "right_hip_pitch_joint", "right_hip_yaw_joint", "right_hip_roll_joint"]
    
        rate = rospy.Rate(60)  # 60Hz update rate
    
        while not rospy.is_shutdown():
            # Update joint positions
            joint_state.position = self.r2c_ahw_msg.q[2:6]+self.r2c_ahw_msg.q[8:12] + self.r2c_leg_msg.q[:12]
            # print("joint",self.r2c_leg_msg.q)
            # Update the timestamp
            joint_state.header.stamp = rospy.Time.now()
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "odom"       # Parent frame
            t.child_frame_id = "base"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = self.r2c_sensor_msg.quat[1]
            t.transform.rotation.y = self.r2c_sensor_msg.quat[2]
            t.transform.rotation.z = self.r2c_sensor_msg.quat[3]
            t.transform.rotation.w = self.r2c_sensor_msg.quat[0]        
            self.br.sendTransform(t)
            # Publish the message
            self.joint_state_ros_pub.publish(joint_state)
            rate.sleep()



    def _init_buffer(self):
        pass

class gamepad_lcm:
    def __init__(self,
                            recv_freq=100,
                            ttl=1,
                            gamepad_topic_name="gamepad2controller",) -> None:
        self.recv_freq = recv_freq
        self._sub_topic = gamepad_topic_name
        self.lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))
        self.gp_msg= gamepad_lcmt()
        self.lcm_receiver_thread = threading.Thread(target=self.thread_2)
        self.s=time.time()
        subscription = self.lcm.subscribe(gamepad_topic_name, self.my_handler)

    def thread_2(self):
        while 1:
            self.lcm.handle()
            time.sleep(1.0/self.recv_freq)

    def my_handler(self,channel, data):
        self.gp_msg = gamepad_lcmt.decode(data)
    def start_recving(self):
        self.lcm_receiver_thread.start()

if __name__=="__main__":
    r_lcm = robot_lcm()
    r_lcm.start_sending()
    r_lcm.start_recving()
