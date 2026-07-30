#include <iostream>
#include <qi/application.hpp>
#include <qi/session.hpp>
#include <qi/os.hpp>

int main(int argc, char* argv[]) {
    std::cout << "[SDK Test] Initializing Aldebaran qi::Application framework..." << std::endl;
    
    // 1. Initialize the core framework application
    qi::Application app(argc, argv);

    std::cout << "[SDK Test] Creating an isolated local session (No Robot Required)..." << std::endl;
    
    // 2. Instantiate a local session
    qi::SessionPtr localSession = qi::makeSession();

    // 3. Test the SDK internal timing mechanism to ensure libqi libraries can run on your OS
    std::cout << "[SDK Test] Testing framework os/time hooks..." << std::endl;
    qi::os::msleep(500); 

    std::cout << "[SDK Test] Successfully verified SDK initialization structures!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << " Result: SUCCESSFUL COMPILE AND SDK LINK! " << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
