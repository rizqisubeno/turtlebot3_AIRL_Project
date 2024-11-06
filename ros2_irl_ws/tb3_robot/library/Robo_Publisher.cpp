#include "Robo_Publisher.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <string>

Robo_Publisher::Robo_Publisher(const std::string& pub_address, const std::string& rep_address)
    : context(1), robo_socket(context, ZMQ_XPUB), rep_context(1), rep_socket(rep_context, ZMQ_REP), running(true),
      can_publish(false), msg_publish(false), msg_is_published(false), success_publish(false), is_state_published(false) {
    // std::cout << "initialize" << std::endl;
    robo_socket.bind(pub_address);
    // std::cout << "initialize1" << std::endl;
    rep_socket.bind(rep_address);
    // std::cout << "initialize2" << std::endl;
}

void Robo_Publisher::start() {
    while (running) {
        receiveSubscriptions();
        if (msg_publish && can_publish) {
            publishMessages();
            msg_is_published = true;
            msg_publish = false;
        }

        if (msg_is_published) {
            zmq::message_t request;
            zmq::recv_result_t result = rep_socket.recv(request, zmq::recv_flags::dontwait);
            if (result) {
                std::string request_str(static_cast<char*>(request.data()), request.size());
                // std::cout << "Robot Received request: " << request_str << std::endl;

                std::string reply;
                if (request_str == "OK") {
                    reply = "OK";
                    msg_publish = false;
                    success_publish = true;
                } else {
                    reply = "FAIL";
                    msg_publish = true;
                    success_publish = false;
                }

                zmq::message_t reply_msg(reply.size());
                memcpy(reply_msg.data(), reply.c_str(), reply.size());
                rep_socket.send(reply_msg, zmq::send_flags::none);

                msg_is_published = false;
                is_state_published = true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Robo_Publisher::stop() {
    running = false;
}

void Robo_Publisher::receiveSubscriptions() {
    zmq::message_t message;
    while (robo_socket.recv(message, zmq::recv_flags::dontwait)) {
        std::string msg(static_cast<char*>(message.data()), message.size());
        char subscriptionType = msg[0];
        std::string topic = msg.substr(1);

        if (subscriptionType == 1) {
            subscriptions[topic]++;
            std::cout << "Subscribed to topic: " << topic << std::endl;
            can_publish = true;
        } else if (subscriptionType == 0) {
            subscriptions[topic]--;
            if (subscriptions[topic] <= 0) {
                subscriptions.erase(topic);
            }
            can_publish = false;
            std::cout << "Unsubscribed from topic: " << topic << std::endl;
        }
    }
}

void Robo_Publisher::publishMessages() {
    for (const auto& sub : subscriptions) {
        if (sub.second > 0) {
            std::string message = sub.first + ": " + encoded_msg;
            zmq::message_t zmqMessage(message.size());
            memcpy(zmqMessage.data(), message.data(), message.size());
            robo_socket.send(zmqMessage, zmq::send_flags::none);
            // std::cout << "Robot encoded msg sent." << std::endl;
        }
    }
}
