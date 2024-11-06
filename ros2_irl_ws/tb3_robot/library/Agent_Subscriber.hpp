#ifndef AGENT_SUBSCRIBER_HPP
#define AGENT_SUBSCRIBER_HPP

#include <string>
#include <zmq.hpp>
#include <message_vec.pb.h>

class Agent_Subscriber {
public:
    Agent_Subscriber(const std::string& sub_address, const std::string& req_address);

    void start();  // Method to start the subscribing start & step loop
    void stop();   // Method to stop the subscribing start & step loop

    std::string encoded_msg;
    bool msg_is_received;
    bool success_received;
    bool is_agent_message_start_received;
    bool is_agent_message_step_received;
    int64_t start_idx;// unsigned long long start_idx;
    int64_t step_idx;// unsigned long long step_idx;

    start_agent_msg start_msg;
    step_agent_msg step_msg;

private:
    zmq::context_t start_context, step_context, req_context;
    zmq::socket_t agent_start_socket, agent_step_socket, req_socket;
    bool running;

    void receive_start_msg();
    void receive_step_msg();
};

#endif // AGENT_SUBSCRIBER_HPP
