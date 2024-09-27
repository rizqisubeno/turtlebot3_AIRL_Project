#include "Agent_Subscriber.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <string>

Agent_Subscriber::Agent_Subscriber(const std::string& sub_address, const std::string& req_address)
    : start_context(1), agent_start_socket(start_context, zmq::socket_type::sub),
      step_context(1), agent_step_socket(step_context, zmq::socket_type::sub),
      req_context(1), req_socket(req_context, zmq::socket_type::req), running(true),
      msg_is_received(false), success_received(false),
      is_agent_message_start_received(false), is_agent_message_step_received(false),
      start_idx(0), step_idx(0) {
    agent_start_socket.connect(sub_address);
    agent_start_socket.set(zmq::sockopt::subscribe, "start");

    agent_step_socket.connect(sub_address);
    agent_step_socket.set(zmq::sockopt::subscribe, "step");

    req_socket.connect(req_address);
}

void Agent_Subscriber::receive_start_msg() {
    zmq::message_t message;
    zmq::recv_result_t result = agent_start_socket.recv(message, zmq::recv_flags::dontwait);

    if (result) {
        std::string received_message(static_cast<char*>(message.data()), message.size());
        if (received_message.starts_with("start")) {
            start_agent_msg agent_msg;
            agent_msg.ParseFromString(received_message.substr(strlen("start") + 2));

            if ((agent_msg.lidar_for_state() != 0) && (agent_msg.agent_state() == "r") && (agent_msg.idx() == start_idx)) {
                std::string reply = "OK";
                zmq::message_t reply_msg(reply.size());
                memcpy(reply_msg.data(), reply.c_str(), reply.size());
                req_socket.send(reply_msg, zmq::send_flags::none);
                start_msg = agent_msg;
                start_idx++;
            } else {
                std::string reply = "FAIL";
                zmq::message_t reply_msg(reply.size());
                memcpy(reply_msg.data(), reply.c_str(), reply.size());
                req_socket.send(reply_msg, zmq::send_flags::none);
            }
            
            bool is_got_rep = false;
            while (!is_got_rep) {
                zmq::message_t request;
                zmq::recv_result_t result = req_socket.recv(request, zmq::recv_flags::dontwait);
                if (result) {
                    std::string request_str(static_cast<char*>(request.data()), request.size());
                    // std::cout << "Agent Received request: " << request_str << std::endl;
                    is_got_rep = true;
                }
            }
            is_agent_message_start_received = true;
        }
    }
}

void Agent_Subscriber::receive_step_msg() {
    zmq::message_t message;
    zmq::recv_result_t result = agent_step_socket.recv(message, zmq::recv_flags::dontwait);

    if (result) {
        std::string received_message(static_cast<char*>(message.data()), message.size());
        if (received_message.starts_with("step")) {
            step_agent_msg agent_msg;
            agent_msg.ParseFromString(received_message.substr(strlen("step") + 2));

            if (agent_msg.idx() == step_idx) {
                std::string reply = "OK";
                zmq::message_t reply_msg(reply.size());
                memcpy(reply_msg.data(), reply.c_str(), reply.size());
                req_socket.send(reply_msg, zmq::send_flags::none);
                step_msg = agent_msg;
                step_idx++;
            } else {
                std::string reply = "FAIL";
                zmq::message_t reply_msg(reply.size());
                memcpy(reply_msg.data(), reply.c_str(), reply.size());
                req_socket.send(reply_msg, zmq::send_flags::none);
            }

            bool is_got_rep = false;
            while (!is_got_rep) {
                zmq::message_t request;
                zmq::recv_result_t result = req_socket.recv(request, zmq::recv_flags::dontwait);
                if (result) {
                    std::string request_str(static_cast<char*>(request.data()), request.size());
                    // std::cout << "Agent Received request: " << request_str << std::endl;
                    is_got_rep = true;
                }
            }
            is_agent_message_step_received = true;
        }
    }
}

void Agent_Subscriber::start() {
    while (running) {
        receive_start_msg();
        receive_step_msg();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Agent_Subscriber::stop() {
    running = false;
}

