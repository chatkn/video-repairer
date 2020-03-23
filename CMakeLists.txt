cmake_minimum_required(VERSION 2.8.11)

project(video-repairer CXX)

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)

find_package(OpenCV REQUIRED )

SET(SOURCES
	src/videoRepairer.cpp
	src/main.cpp
)

SET(HEADER
	include/videoRepairer.hh
)


include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${OpenCV_INCLUDE_DIRS}
  )

link_directories(
	${CMAKE_BINARY_DIR}/bin
	${OpenCV_LIB_DIR}
	)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADER})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#debug opencv_world340d optimized opencv_world340)

INSTALL(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
