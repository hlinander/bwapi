
include_directories(
  ../include
  ../ExampleAIModule/Source/cereal-1.2.2/include
  ../ExampleAIModule/Source/eigen
  ../ExampleAIModule/Source/
)

set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_executable(Overmind
  Overmind.cpp
  ../ExampleAIModule/Source/AIStructs.cpp
  )

target_link_libraries(Overmind "${TORCH_LIBRARIES}")
set_property(TARGET Overmind PROPERTY CXX_STANDARD 14)