cmake_minimum_required(VERSION 3.10.2)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
project(SurfaceCompare)

# Find VTK
find_package(VTK COMPONENTS 
  CommonColor
  CommonCore
  CommonDataModel
  CommonTransforms
  FiltersCore
  FiltersPoints
  FiltersGeneral
  FiltersExtraction
  FiltersGeometry
  FiltersModeling
  FiltersSources
  IOGeometry
  IOLegacy
  IOPLY
  IOXML
  InteractionStyle
  InteractionWidgets
  RenderingAnnotation
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
 )
add_executable(SurfaceCompare SurfaceCompare.cxx)

target_link_libraries(SurfaceCompare PRIVATE ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS SurfaceCompare
  MODULES ${VTK_LIBRARIES})