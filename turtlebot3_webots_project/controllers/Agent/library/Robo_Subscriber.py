import zmq
from library.message_vec_pb2 import robot_msg
from types import SimpleNamespace

class RobotSubscriber: 
    def __init__(self, robot_socket="tcp://127.0.0.1:5555", req_socket="tcp://127.0.0.1:6666"):
        self.context = zmq.Context()
        self.req_context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.req_socket = self.req_context.socket(zmq.REQ)

        self.socket.connect(robot_socket)
        self.req_socket.connect(req_socket)
        self.topic = "state"
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self.msg = robot_msg()  
        self.states = None
        self.idx = 0
        self.is_ok = False

    def check_and_parse_msg(self, verbose=False):
        lines = str(self.msg).splitlines()
        state_dict = {}
        lidar_dat = []

        for item in lines:
            key, value = item.split(': ')
            if key == "lidar_data":
                lidar_dat.append(float(value))
            else:
                state_dict[key] = float(value) if '.' in value else int(value)

        state_dict['lidar_data'] = lidar_dat
        state = SimpleNamespace(**state_dict)
        if verbose:
            for key, val in state_dict.items():
                print(f"{key} :\t{val}")

        return state

    def get_msg(self, non_block=False, verbose=False):
        encoded_msg = None
        if non_block:
            try:
                encoded_msg = self.socket.recv(zmq.NOBLOCK)
                # print(f"Received message")
            except zmq.Again:
                # print("No message received.")
                pass
        else:
            encoded_msg = self.socket.recv()

        self.msg.ParseFromString(encoded_msg[(len(self.topic)+2):])
        self.states = self.check_and_parse_msg(verbose=verbose)

        msg = None
        if self.states and self.idx == self.states.idx:
            msg = "OK"
            self.is_ok = True
        else:
            msg = "FAIL"
            self.is_ok = False

        self.idx += 1

        req_success = False
        while not req_success:
            try:
                self.req_socket.send_string(msg, flags=zmq.DONTWAIT)
                reply = self.req_socket.recv_string()
                # print(f"Received reply: {reply}")
                if reply == msg:
                    req_success = True
            except zmq.ZMQError:
                pass

        return self.is_ok

