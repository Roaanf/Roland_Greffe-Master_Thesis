/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "GigaSpace", "index.html", [
    [ "Overview", "index.html#Intro_Section", null ],
    [ "What is GigaSpace / GigaVoxels ?", "index.html#section_Intro_Section_WhatIsGigaSpaceGigaVoxels", [
      [ "GigaVoxels", "index.html#subsection_Intro_Section_GigaVoxels", null ],
      [ "GigaSpace", "index.html#subsection_Intro_Section_GigaSpace", null ],
      [ "Cuda vs OpenGL vs GLSL", "index.html#subsection_Intro_Section_CudaVsOpenGLVsGLSL", null ],
      [ "History and roots of Gigavoxels", "index.html#subsection_Intro_Section_HistoryAndRootsOfGigavoxels", null ]
    ] ],
    [ "Programming the GigaVoxels library", "index.html#library_ProgrammingTheGigaVoxelsLibrary", null ],
    [ "Research Team", "index.html#Research_Team_section", null ],
    [ "Authors", "authors.html", [
      [ "The Team", "authors.html#authors_section", null ],
      [ "Contact", "authors.html#authors_contact", null ]
    ] ],
    [ "Programming the Library", "_programming_the_giga_voxels_library.html", [
      [ "Technologies", "_programming_the_giga_voxels_library.html#Technologies_Section", null ],
      [ "Mecanismns", "_programming_the_giga_voxels_library.html#library_Mecanismn", null ],
      [ "Programmation", "_programming_the_giga_voxels_library.html#library_Programmation", null ],
      [ "Tutorials", "_programming_the_giga_voxels_library.html#tutorials_section", null ],
      [ "Tools", "_programming_the_giga_voxels_library.html#tools_section", null ],
      [ "External Libraries", "_programming_the_giga_voxels_library.html#library_ExternalLibraries", null ],
      [ "Mecanismns", "_mecanismns.html", [
        [ "The GigaVoxels Approach", "_mecanismn__the_giga_voxels_approach.html", [
          [ "References and Publications", "_mecanismn__the_giga_voxels_approach.html#Mecanismn_TheGigaVoxelsApproach_References_Section", [
            [ "Publications", "_mecanismn__the_giga_voxels_approach.html#Publications", null ],
            [ "Videos", "_mecanismn__the_giga_voxels_approach.html#Videos", null ]
          ] ],
          [ "The GigaVoxels Pipeline", "_mecanismn__the_giga_voxels_approach.html#Mecanismn_TheGigaVoxelsPipieline_Section", [
            [ "Overview", "_mecanismn__the_giga_voxels_approach.html#Overview", null ],
            [ "Pre-Render Pass", "_mecanismn__the_giga_voxels_approach.html#subsection_Mecanismn_TheGigaVoxelsApproach_PreRenderPass", null ],
            [ "Rendering Pass", "_mecanismn__the_giga_voxels_approach.html#subsection_Mecanismn_TheGigaVoxelsApproach_RenderingPass", null ],
            [ "Post-Render Pass", "_mecanismn__the_giga_voxels_approach.html#subsection_Mecanismn_TheGigaVoxelsApproach_PostRenderPass", null ]
          ] ],
          [ "Detailed Pipeline's Sequence Diagram of a GPU Producer", "_mecanismn__the_giga_voxels_approach.html#Mecanismn_Mecanismn_TheGigaVoxelsApproach_GPUProducerDetailedSequence_Section", null ]
        ] ],
        [ "I/O Streaming and File Formats", "_mecanismn__i_o__streaming.html", [
          [ "Overview", "_mecanismn__i_o__streaming.html#Overview_Section", null ],
          [ "I/O Streaming Mecanismn", "_mecanismn__i_o__streaming.html#FileFormatsMecanismn_Section", null ],
          [ "File Formats and Organization on disk", "_mecanismn__i_o__streaming.html#FileFormatsOnDiskOrganization_Section", [
            [ "Description", "_mecanismn__i_o__streaming.html#Description", null ],
            [ "Example", "_mecanismn__i_o__streaming.html#Example", null ]
          ] ],
          [ "Tools", "_mecanismn__i_o__streaming.html#FileFormatsTools_Section", [
            [ "Voxelizer", "_mecanismn__i_o__streaming.html#Voxelizer", null ]
          ] ]
        ] ]
      ] ],
      [ "Programmation", "_library__how_to_program.html", [
        [ "Introduction", "_library__how_to_program.html#Library_HowToProgram_Introduction_Section", null ],
        [ "Overview", "_library__how_to_program.html#Library_HowToProgram_Overview_Section", null ],
        [ "List of Modules", "_library__how_to_program.html#Library_HowToProgram_Modules_Section", [
          [ "Core Module", "_library__how_to_program.html#subsection_Core_Module", null ],
          [ "Structure Module", "_library__how_to_program.html#subsection_Structure_Module", null ],
          [ "Cache Module", "_library__how_to_program.html#subsection_Cache_Module", null ],
          [ "Rendering Module", "_library__how_to_program.html#subsection_Rendering_Module", null ],
          [ "Performance Monitoring Module", "_library__how_to_program.html#subsection_PerformanceMonitoring_Module", null ],
          [ "Utils Module", "_library__how_to_program.html#subsection_Utils_Module", null ]
        ] ],
        [ "Technologies", "_library__how_to_program__technologies__page.html", [
          [ "Overview", "_library__how_to_program__technologies__page.html#section_HowToProgram_Technologies_Overview", null ],
          [ "GPU template metaprogrammation", "_library__how_to_program__technologies__page.html#section_HowToProgram_Technologies_GPUTemplateProgramming", null ]
        ] ],
        [ "Prerequisites", "_library__how_to_program__prerequisites__page.html", [
          [ "CUDA", "_library__how_to_program__prerequisites__page.html#section_Library_HowToProgram_Prerequisites_CUDA", [
            [ "Prerequisites", "_library__how_to_program__prerequisites__page.html#Prerequisites", null ]
          ] ]
        ] ],
        [ "Programmation : The Basics", "_library__how_to_program__the_basics__page.html", [
          [ "Introduction", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_Introduction", null ],
          [ "GigaVoxels Key Features", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_GigaVoxelsKeyFeatures", null ],
          [ "Data Production Management", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_DataProductionManagement", [
            [ "Node subdivision", "_library__how_to_program__the_basics__page.html#subsection_Library_HowToProgram_TheBasics_NodeSubdivision", null ],
            [ "Brick production", "_library__how_to_program__the_basics__page.html#subsection_Library_HowToProgram_TheBasics_BrickProduction", null ],
            [ "Rendering Stage", "_library__how_to_program__the_basics__page.html#subsection_Library_HowToProgram_TheBasics_RenderingStage", null ]
          ] ],
          [ "Writing a GPU Producer", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_ProducerWriting", [
            [ "Overview", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_ProducerWriting_Overview", null ],
            [ "Class Definition", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_ProducerWriting_Definition", null ],
            [ "Class Implementation", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_ProducerWriting_Implementation", null ]
          ] ],
          [ "Writing a Shader", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_ShaderWriting", [
            [ "Class Definition", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_ShaderWriting_Definition", null ],
            [ "Class Implementation", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_ShaderWriting_Implementation", null ]
          ] ],
          [ "Helper classes and functions", "_library__how_to_program__the_basics__page.html#section_Library_HowToProgram_TheBasics_HelperClassesAndFunctions", [
            [ "Common Host Producer", "_library__how_to_program__the_basics__page.html#Using", null ]
          ] ]
        ] ]
      ] ],
      [ "Tutorials", "_tutorials.html", [
        [ "Types of tutorials", "_tutorials.html#Type_Section", null ],
        [ "I/O Streaming Tutorials", "_tutorials__i_o__streaming.html", [
          [ "Dynamic Load Tutorial Dynamic Load Tutorial", "_tutorial__dynamic__load.html", [
            [ "Recommandation", "_tutorial__dynamic__load.html#Tutorial_Dynamic_Load_Mandatory_Section", null ],
            [ "Data Structure", "_tutorial__dynamic__load.html#Tutorial_Dynamic_Load_DataStructure", null ],
            [ "Node tiles production", "_tutorial__dynamic__load.html#Tutorial_Dynamic_Load_Node_Tiles_Production", null ],
            [ "Brick of voxels production", "_tutorial__dynamic__load.html#Tutorial_Dynamic_Load_Brick_Of_Voxels", null ],
            [ "Node tiles production on HOST", "_tutorial__dynamic__load.html#Tutorial_Dynamic_Load_Node_Tiles_Production_on_HOST", null ],
            [ "Brick of voxels production on HOST", "_tutorial__dynamic__load.html#Tutorial_Dynamic_Load_Brick_Of_Voxels_on_HOST", null ]
          ] ]
        ] ],
        [ "Procedural Geometry Tutorials", "_tutorials__procedural__geometry.html", [
          [ "List of tutorials", "_tutorials__procedural__geometry.html#List_Section", null ],
          [ "Simple Sphere Tutorial", "_tutorial__simple__sphere.html", [
            [ "Recommandation", "_tutorial__simple__sphere.html#Tutorial_Simple_Sphere_Mandatory_Section", null ],
            [ "UML Design", "_tutorial__simple__sphere.html#UMLDesign_Section", null ],
            [ "Data Structure", "_tutorial__simple__sphere.html#Tutorial_Simple_Sphere_DataStructure", null ],
            [ "Node tiles production", "_tutorial__simple__sphere.html#Tutorial_Simple_Sphere_Node_Tiles_Production", null ],
            [ "Brick of voxels production", "_tutorial__simple__sphere.html#Tutorial_Simple_Sphere_Brick_Of_Voxels", null ],
            [ "N-Tree Visualization", "_tutorial__simple__sphere.html#NTree_Section", null ]
          ] ],
          [ "Simple Sphere CPU Tutorial", "_tutorial__simple__sphere__c_p_u.html", null ],
          [ "Mandelbrot Set Tutorial", "_tutorial__mandelbrot__set.html", null ],
          [ "Slisesix Tutorial", "_tutorial__slisesix.html", [
            [ "Screenshot", "_tutorial__slisesix.html#Screenshot_Section", null ]
          ] ]
        ] ],
        [ "Noise Tutorials", "_tutorials__noise.html", [
          [ "Amplified Surface Tutorial", "_tutorial__amplified__surface.html", null ],
          [ "Amplified Volume Tutorial", "_tutorial__amplified__volume.html", null ],
          [ "Smart Perlin Noise Tutorial", "_tutorial__smart__perlin__noise.html", null ],
          [ "Noise Inheritance Tutorial", "_tutorial__noise__inheritance.html", null ],
          [ "Procedural Terrain Tutorial", "_tutorial__procedural__terrain.html", null ]
        ] ],
        [ "3D Effects Tutorials", "_tutorials_3_d__effects.html", [
          [ "Depth of Field Tutorial", "_tutorial__depth__of__field.html", null ]
        ] ],
        [ "Geometry Instancing Tutorials", "_tutorials__geometry__instancing.html", [
          [ "Menger Sponge Tutorial", "_tutorial__menger__sponge.html", null ]
        ] ],
        [ "GUI Integration Tutorials", "_tutorials__g_u_i__integration.html", [
          [ "GLUT Window Tutorial", "_tutorial__g_l_u_t__window.html", null ]
        ] ],
        [ "Graphics Interoperability Tutorials", "_tutorials__graphics__interoperability.html", [
          [ "Renderer GLSL Tutorial", "_tutorial__renderer__g_l_s_l.html", null ],
          [ "Renderer GLSL Sphere Tutorial", "_tutorial__renderer__g_l_s_l__sphere.html", null ],
          [ "Proxy Geometry Tutorial", "_tutorial__proxy__geometry.html", null ],
          [ "Voxelization Tutorial", "_tutorial__voxelization.html", [
            [ "Recommandation", "_tutorial__voxelization.html#Tutorial_Voxelization_Mandatory_Section", null ]
          ] ],
          [ "Signed Distance Field Voxelization Tutorial", "_tutorial__voxelization__signed__distance__field.html", [
            [ "Recommandation", "_tutorial__voxelization__signed__distance__field.html#Tutorial_Voxelization_Signed_Distance_Field_Mandatory_Section", null ]
          ] ]
        ] ]
      ] ],
      [ "Tools", "_tools.html", [
        [ "GigaVoxels Viewer", "_tool__gv_viewer.html", [
          [ "Dependances", "_tool__gv_viewer.html#Dependances_Section", null ],
          [ "Features", "_tool__gv_viewer.html#Features_Section", null ],
          [ "Philosophy", "_tool__gv_viewer.html#Philosophy_Section", null ]
        ] ],
        [ "GigaVoxels Voxelizer", "_tool__gv_voxelizer.html", [
          [ "Design", "_tool__gv_voxelizer.html#voxelizer_UML_Design", null ]
        ] ],
        [ "GigaVoxels DICOM Voxelizer", "_tool__gv_dicom_voxelizer.html", null ]
      ] ],
      [ "External Libraries", "_external_libraries.html", [
        [ "List of external libraries", "_external_libraries.html#ExternalLibrariesList_Section", [
          [ "CUDA", "_external_libraries.html#subsection_CUDA", null ],
          [ "Thrust", "_external_libraries.html#subsection_Thrust", null ],
          [ "CUDPP", "_external_libraries.html#subsection_CUDPP", null ],
          [ "Loki", "_external_libraries.html#subsection_Loki", null ],
          [ "Assimp", "_external_libraries.html#subsection_Assimp", null ],
          [ "Qt", "_external_libraries.html#subsection_Qt", null ],
          [ "QGLViewer", "_external_libraries.html#subsection_QGLViewer", null ],
          [ "CMake", "_external_libraries.html#subsection_CMake", null ]
        ] ]
      ] ]
    ] ],
    [ "Todo List", "todo.html", null ],
    [ "Topics", "topics.html", "topics" ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Typedefs", "functions_type.html", null ],
        [ "Enumerations", "functions_enum.html", null ],
        [ "Enumerator", "functions_eval.html", null ],
        [ "Related Symbols", "functions_rela.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", "globals_dup" ],
        [ "Functions", "globals_func.html", null ],
        [ "Variables", "globals_vars.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Enumerator", "globals_eval.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"_external_libraries.html",
"_gs_i_data_loader_8inl.html",
"_gs_renderer_c_u_d_a_kernel_8h.html#ab4e3ba1972c75bbcd1535808a1a38eb1",
"_library__how_to_program__the_basics__page.html",
"class_gs_graphics_1_1_gs_shader_program.html#a09c942f140582bcb61d82e0890d7311a",
"class_gv_cache_1_1_gs_g_l_cache_manager.html#accc4c91a190ad8a743b629d7eb844823a5741fbdf7dee0b1456dac786d699d2a9",
"class_gv_core_1_1_gs_provider.html#ade22afd88d522e28ca5cc0b92af23ebf",
"class_gv_rendering_1_1_gs_graphics_resource.html#a768af92d2b1965327bddd7574b1ac23ca9f0b86e52c140de4daefbde4cc0f6c02",
"class_gv_rendering_1_1_gs_renderer_c_u_d_a.html#ac463f9c9fd2251b5f6c7c6f4c8ad51e2",
"class_gv_structure_1_1_gs_i_writer.html#a9b4d4408312a6bea2f12fa5dca635335",
"class_gv_utils_1_1_gs_file_name_builder.html#a03bb64928275d64bd6d9745b3b190d61",
"class_gv_utils_1_1_gs_ray_map.html#aee518cd9c962548d86ddd121f314d404a9b52fff46e6216cc7de2735e4c0503ee",
"class_gv_utils_1_1_gv_pre_integrated_transfer_function.html#a9927f22c0f2dcb2e877f7de0f38c1465",
"functions_func_i.html",
"struct_gv_core_1_1_gs_localization_info.html#a172de7f058a40af5600825f88a839b83",
"struct_gv_structure_1_1_gs_production_statistics.html#ad70b39f2aa920dcb197312e4aab01d28"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';