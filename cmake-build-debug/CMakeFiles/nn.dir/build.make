# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yuqiongli/Desktop/GPU/gpu-exercises

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yuqiongli/Desktop/GPU/gpu-exercises/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/nn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/nn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nn.dir/flags.make

CMakeFiles/nn.dir/nn.cpp.o: CMakeFiles/nn.dir/flags.make
CMakeFiles/nn.dir/nn.cpp.o: ../nn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yuqiongli/Desktop/GPU/gpu-exercises/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nn.dir/nn.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn.dir/nn.cpp.o -c /Users/yuqiongli/Desktop/GPU/gpu-exercises/nn.cpp

CMakeFiles/nn.dir/nn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn.dir/nn.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yuqiongli/Desktop/GPU/gpu-exercises/nn.cpp > CMakeFiles/nn.dir/nn.cpp.i

CMakeFiles/nn.dir/nn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn.dir/nn.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yuqiongli/Desktop/GPU/gpu-exercises/nn.cpp -o CMakeFiles/nn.dir/nn.cpp.s

CMakeFiles/nn.dir/nn.cpp.o.requires:

.PHONY : CMakeFiles/nn.dir/nn.cpp.o.requires

CMakeFiles/nn.dir/nn.cpp.o.provides: CMakeFiles/nn.dir/nn.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn.dir/build.make CMakeFiles/nn.dir/nn.cpp.o.provides.build
.PHONY : CMakeFiles/nn.dir/nn.cpp.o.provides

CMakeFiles/nn.dir/nn.cpp.o.provides.build: CMakeFiles/nn.dir/nn.cpp.o


# Object files for target nn
nn_OBJECTS = \
"CMakeFiles/nn.dir/nn.cpp.o"

# External object files for target nn
nn_EXTERNAL_OBJECTS =

nn: CMakeFiles/nn.dir/nn.cpp.o
nn: CMakeFiles/nn.dir/build.make
nn: CMakeFiles/nn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yuqiongli/Desktop/GPU/gpu-exercises/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable nn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nn.dir/build: nn

.PHONY : CMakeFiles/nn.dir/build

CMakeFiles/nn.dir/requires: CMakeFiles/nn.dir/nn.cpp.o.requires

.PHONY : CMakeFiles/nn.dir/requires

CMakeFiles/nn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nn.dir/clean

CMakeFiles/nn.dir/depend:
	cd /Users/yuqiongli/Desktop/GPU/gpu-exercises/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yuqiongli/Desktop/GPU/gpu-exercises /Users/yuqiongli/Desktop/GPU/gpu-exercises /Users/yuqiongli/Desktop/GPU/gpu-exercises/cmake-build-debug /Users/yuqiongli/Desktop/GPU/gpu-exercises/cmake-build-debug /Users/yuqiongli/Desktop/GPU/gpu-exercises/cmake-build-debug/CMakeFiles/nn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nn.dir/depend
