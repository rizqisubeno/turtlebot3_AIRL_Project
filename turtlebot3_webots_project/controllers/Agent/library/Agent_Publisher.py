import zmq
from library.message_vec_pb2 import start_agent_msg, step_agent_msg

class AgentPublisher: 
    def __init__(self, agent_socket="tcp://127.0.0.1:7777", rep_socket="tcp://127.0.0.1:8888"):
        self.context = zmq.Context(1)
        self.socket = self.context.socket(zmq.XPUB)
        self.socket.bind(agent_socket)
        self.start_msg = start_agent_msg()
        self.step_msg = step_agent_msg()
        self.start_idx = 0
        self.step_idx = 0

        self.rep_context = zmq.Context(1)
        self.rep_socket = self.rep_context.socket(zmq.REP)
        self.rep_socket.bind(rep_socket)
        
        self.subscriptions = {}
        self.is_got_reply = False

    def receiveSubscriptions(self):
        try:
            message = self.socket.recv(zmq.DONTWAIT)
            subscription_type = message[0]
            topic = message[1:].decode('utf-8')

            if subscription_type == 1:
                self.subscriptions[topic] = self.subscriptions.get(topic, 0) + 1
                print(f"Subscribed to topic: {topic}")
            elif subscription_type == 0:
                self.subscriptions[topic] = self.subscriptions.get(topic, 1) - 1
                if self.subscriptions[topic] <= 0:
                    del self.subscriptions[topic]
                print(f"Unsubscribed from topic: {topic}")
        except zmq.Again:
            pass

    def send_start_msg(self, non_block=False, verbose=False):
        serialized_message = self.start_msg.SerializeToString()
        msg = "start: ".encode('utf-8') + serialized_message
        if non_block:
            self.socket.send(msg, flags=zmq.DONTWAIT)
        else:
            self.socket.send(msg)

    def send_step_msg(self, non_block=False, verbose=False):
        serialized_message = self.step_msg.SerializeToString()
        msg = "step: ".encode('utf-8') + serialized_message
        if non_block:
            self.socket.send(msg, flags=zmq.DONTWAIT)
        else:
            self.socket.send(msg)

    def run(self, mode=None):
        is_got_reply = False
        is_message_sended = False
        toggle_message_send = True
        
        while not is_got_reply:
            if toggle_message_send and not is_message_sended and self.subscriptions:
                if mode == "start":
                    self.send_start_msg()
                elif mode == "step":
                    self.send_step_msg()
                is_message_sended = True
                toggle_message_send = False

            if is_message_sended:
                try:
                    msg = self.rep_socket.recv_string(zmq.DONTWAIT)
                    if msg == "OK":
                        self.rep_socket.send_string(msg)
                        is_got_reply = True
                    elif msg == "FAIL":
                        self.rep_socket.send_string(msg)
                        is_got_reply = False
                        is_message_sended = False
                        toggle_message_send = True
                except zmq.Again:
                    is_got_reply = False

        self.is_got_reply = is_got_reply

