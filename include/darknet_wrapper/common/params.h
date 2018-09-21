#pragma once

#include "yaml-cpp/yaml.h"
#include <string>
#include <vector>
#include <exception>

namespace darknet_wrapper { namespace common {

/** \class Params
* \brief Helper class to load and access static parameters 
*        from param/static_default\.yaml. 
*/

class Params { 

public:

  Params() = default;

  /**
  * \brief Loads all of the parameters from the indicated yaml file.
  * @param filename Full path to the yaml file.
  */
  bool Initialize(std::string filename)
  {
    bool read_parameters = false;
    try 
    {

      config_ = YAML::LoadFile(filename);

      read_parameters = true;
    }
    catch (std::exception& e)
    {
      std::cout << e.what() <<std::endl;
      std::cout << "YoloObjectDetector: Param filename not found: " <<filename <<std::endl;
    }

    return read_parameters;
  }

  /**
  * \brief Accesor to the parameters in the yaml file.
  * @param key The name of the parameter.
  * @param value The output value of the parameter.
  * @param usr_default User supplied default value incase the parameter isn't found.
  * @return Return true if the parameter was found.
  */
  template<class T, class U>
  bool GetParam(const std::string key, T& value, U usr_default) const
  {
    bool parameter_retrieved = false;

    // If the parameter exist, return the value.
    // Else, return the user default value.
    if (config_[key])
    {
      value = config_[key].as<T>();
      parameter_retrieved = true;

    } 
    else
      value = static_cast<T>(usr_default);

    return parameter_retrieved;
  }

  
  YAML::Node config_; /**< Container for all of the parameters */


};
}}