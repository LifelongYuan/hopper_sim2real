import lcm
from hopper_lcm_types.lcm_types.python.hopper_data_lcmt import *
from hopper_lcm_types.lcm_types.python.hopper_cmd_lcmt import *
from hopper_lcm_types.lcm_types.python.gamepad_lcmt import *
from hopper_lcm_types.lcm_types.python.hopper_imu_lcmt import *
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
                pub_topic_name="hopper_cmd",
                sub_topic_name_dict={"data":"hopper_data",
                                    "imu":"hopper_imu",
                                    "gamepad":"gamepad"},
                joint_state_ros_pub=True) -> None:
        self.pub_freq=publish_freq
        self._pub_topic_name = pub_topic_name
        self._sub_name_dict:dict = sub_topic_name_dict
        # filter dict by keys:
        self.filter()

        # clients
        self.r2c = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))
        self.c2r = lcm.LCM("udpm://239.255.76.67:7667?ttl="+str(ttl))

        # msgs - command to send
        self.c2r_cmd_msg = hopper_cmd_lcmt()

        # msgs - received data
        self.r2c_data_msg = hopper_data_lcmt()
        self.r2c_imu_msg = hopper_imu_lcmt()
        self.r2c_gamepad_msg = gamepad_lcmt()

        # TODO: two thread is enough?
        self.lcm_sender_thread = threading.Thread(target=self.thread_1)
        self.lcm_receiver_thread = threading.Thread(target=self.thread_2)
        self.next_publish_time = time.perf_counter()

        if joint_state_ros_pub and not ON_BOARD:
 
            rospy.init_node("hopper_joint_state_publisher", anonymous=True)
            self.joint_state_ros_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
            self.ros_joint_state_pub_thread = threading.Thread(target=self.thread_3)
            self.br = tf2_ros.TransformBroadcaster()
            self.ros_joint_state_pub_thread.start()

        # register handlers
        for key,t_name in self._sub_name_dict.items():
            self.r2c.subscribe(t_name, getattr(self,key+"_handler"))
        self._init_buffer()

    # handlers:
    def data_handler(self, channel, data):
        self.r2c_data_msg = hopper_data_lcmt.decode(data)
    
    def imu_handler(self, channel, data):
        self.r2c_imu_msg = hopper_imu_lcmt.decode(data)
    
    def gamepad_handler(self, channel, data):
        self.r2c_gamepad_msg = gamepad_lcmt.decode(data)

    def filter(self):
        """
        filter unused channels by keys
        """
        pass

    def publish(self):
        self.c2r.publish(self._pub_topic_name, self.c2r_cmd_msg.encode())
    
    def start_sending(self):
        self.lcm_sender_thread.start()

    def start_recving(self):
        self.lcm_receiver_thread.start()

    def thread_1(self):
        period = 1.0 / self.pub_freq
        while 1:
            # Publish the message
            self.publish()
            # Calculate next wake time
            self.next_publish_time += period
            # Sleep until next wake time (drift-free approach)
            sleep_time = self.next_publish_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're behind schedule, don't sleep but update next time
                # to prevent accumulating drift
                self.next_publish_time = time.perf_counter() + period

    def thread_2(self):
        while 1:
            self.r2c.handle()

    def thread_3(self):
        pass


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
