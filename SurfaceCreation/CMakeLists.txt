cmake_minimum_required(VERSION 3.10.2)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
project(SurfaceCreation)

# Find VTK
find_package(VTK COMPONENTS 
  CommonColor
  CommonCore
  CommonDataModel
  FiltersPoints
  FiltersSources
  IOGeometry
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
  PoissonReconstruction
  Powercrust
  IOLegacy
  IOPLY
  IOXML
  CommonTransforms
  FiltersCore
  FiltersGeneral
  ImagingHybrid
 )
add_executable(SurfaceCreation SurfaceCreation.cxx)

target_link_libraries(SurfaceCreation PRIVATE ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS SurfaceCreation
  MODULES ${VTK_LIBRARIES})