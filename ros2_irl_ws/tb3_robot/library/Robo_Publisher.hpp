#ifndef ROBO_PUBLISHER_HPP
#define ROBO_PUBLISHER_HPP

#include <string>
#include <unordered_map>
#include <zmq.hpp>

class Robo_Publisher {
public:
    Robo_Publisher(const std::string& pub_address, const std::string& rep_address);
    void receiveSubscriptions();

    void start();  // Method to start the publishing loop in a separate thread
    void stop();   // Method to stop the publishing loop

    std::string encoded_msg;
    bool can_publish;
    bool msg_publish;
    bool msg_is_published;
    bool success_publish;
    bool is_state_published;
    std::unordered_map<std::string, int> subscriptions;

private:
    zmq::context_t context, rep_context;
    zmq::socket_t robo_socket, rep_socket;
    bool running;

    void publishMessages();
};

#endif // ROBO_PUBLISHER_HPP
