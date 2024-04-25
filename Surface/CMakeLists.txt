cmake_minimum_required(VERSION 3.10.2)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
project(SurfaceDetect)

# Find ITK
find_package(ITK REQUIRED)
find_package(
    VTK COMPONENTS 
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

include(${ITK_USE_FILE})

add_executable(SurfaceDetect SurfaceDetect.cxx)

target_link_libraries(SurfaceDetect PRIVATE ${ITK_LIBRARIES} ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS SurfaceDetect
  MODULES ${VTK_LIBRARIES}
)
