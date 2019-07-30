
include_directories(
  ../include
  Source/cereal-1.2.2/include
  Source/eigen
)

set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(ExampleAIModule SHARED
  Source/Dll.cpp
  Source/ExampleAIModule.cpp
  Source/AIStructs.cpp
  )

target_link_libraries(ExampleAIModule BWAPILIB)
target_link_libraries(ExampleAIModule "${TORCH_LIBRARIES}")
set_property(TARGET ExampleAIModule PROPERTY CXX_STANDARD 14)
