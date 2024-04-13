#include <vtkActor.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkBooleanOperationPolyDataFilter.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPCANormalEstimation.h>
#include <vtkSignedDistance.h>
#include <vtkPoissonReconstruction.h>
#include <vtkPowerCrustSurfaceReconstruction.h>
#include <vtkSurfaceReconstructionFilter.h>
#include <vtkContourFilter.h>
#include <vtkReverseSense.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkRenderer.h>
#include <vtkExtractSurface.h>
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkCylinder.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkClipPolyData.h>
#include <vtkImplicitBoolean.h>
#include <vtkStaticPointLocator.h>
#include <vtkSurfaceNets3D.h>
#include <sstream>

int main(int argc, char* argv[])
{
    bool computePointError = false;
    std::string answer;
    std::cout << "Compute error ?" << std::endl;
    std::cin >> answer;
    if (answer != "0") {
        computePointError = true;
    }

    // TestChamber
    if (true) {

        return 0;
    }

    vtkNew<vtkNamedColors> colors;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);
    renderWindow->SetWindowName("SurfaceCreation");
    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    renderWindowInteractor->SetRenderWindow(renderWindow);
    /* Construct initial model
    std::vector<std::string> files;
    std::string box = "./Data/box.stl";
    std::string cylinder = "./Data/cylinder.stl";
    std::string pyramide = "./Data/pyramide.stl";
    files.push_back(box);
    files.push_back(cylinder);
    files.push_back(pyramide);

    std::vector<vtkSmartPointer<vtkPolyData>> polyDataVec;

    for (const std::string &i : files) {
        vtkNew<vtkSTLReader> reader;
        reader->SetFileName(i.c_str());
        reader->Update();
        auto poly_data = vtkSmartPointer<vtkPolyData>::New();
        poly_data->ShallowCopy(reader->GetOutput());
        polyDataVec.push_back(poly_data);

        // Visualize
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(reader->GetOutputPort());

        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetDiffuse(0.8);
        actor->GetProperty()->SetDiffuseColor(
            colors->GetColor3d("LightSteelBlue").GetData());
        actor->GetProperty()->SetSpecular(0.3);
        actor->GetProperty()->SetSpecularPower(60.0);
        renderer->AddActor(actor);
    }
    auto addFilter = vtkSmartPointer<vtkBooleanOperationPolyDataFilter>::New();
    addFilter->SetInputData(0, polyDataVec[0]);
    addFilter->SetInputData(1, polyDataVec[2]);
    addFilter->SetOperationToUnion();
    addFilter->Update();
    
    auto diffFilter = vtkSmartPointer<vtkBooleanOperationPolyDataFilter>::New();
    diffFilter->SetInputData(0, addFilter->GetOutput());
    diffFilter->SetInputData(1, polyDataVec[1]);
    diffFilter->SetOperationToDifference();
    diffFilter->Update();

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(diffFilter->GetOutputPort());

    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);
    actor->GetProperty()->SetDiffuse(0.8);
    actor->GetProperty()->SetDiffuseColor(colors->GetColor3d("LightSteelBlue").GetData());
    actor->GetProperty()->SetSpecular(0.3);
    actor->GetProperty()->SetSpecularPower(60.0);
    renderer->AddActor(actor);
    */
    std::string filename = "./Data/PointCloud(Maison)Crop.txt";
    std::ifstream filestream(filename.c_str());
    std::string line;
    vtkNew<vtkPoints> points;

    while (std::getline(filestream, line)){
        double x, y, z;
        std::stringstream linestream;
        linestream << line;
        linestream >> x >> y >> z;
        points->InsertNextPoint(x, y, z);
    }

    filestream.close();

    vtkNew<vtkPolyData> polyData;

    polyData->SetPoints(points);

    if (computePointError) {
        // Length of edges
        vtkNew<vtkStaticPointLocator> pointLocator;
        pointLocator->SetDataSet(polyData);
        pointLocator->BuildLocator();

        // Hardcoded array because flemme
        double point1Coord[16][3] = { {-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},
                                      {20.0, 20.0, 20.0},{20.0, 20.0, 20.0},{20.0, 20.0, -20.0},
                                      {20.0, 20.0, -20.0},{-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},
                                      {-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},{20.0, -20.0, 20.0},
                                      {0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0} };

        double point2Coord[16][3] = { {20.0, 20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, 20.0, -20.0},
                                      {20.0, 20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
                                      {-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},
                                      {20.0, -20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
                                      {-20.0, 20.0, 20.0},{-20.0, 20.0, -20.0},{20.0, 20.0, -20.0},{20.0, 20.0, 20.0} };
        double distancesError[16];
        // Loop through all 16 outer edges
        for (size_t i = 0; i < 16; i++){
            vtkIdType point1ID = pointLocator->FindClosestPoint(point1Coord[i]);
            double* point1Point = polyData->GetPoint(point1ID);
            double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
            vtkIdType point2ID = pointLocator->FindClosestPoint(point2Coord[i]);
            double* point2Point = polyData->GetPoint(point2ID);
            double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
            double distance = std::sqrt(vtkMath::Distance2BetweenPoints(point1, point2));
            //compute distance btw the two points
            if (i < 12) {
                distancesError[i] = abs(distance - 40.0);
            } else {
                distancesError[i] = abs(distance - 41.2309);
            }
            std::cout << distancesError[i] << " ";
        }
        std::cout << std::endl;
        // Angles

    }

    // Taken from ExtractSurface example
    double bounds[6];
    polyData->GetBounds(bounds);
    double range[3];
    int sampleSize = 20;
    std::cout << sampleSize << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        range[i] = bounds[2 * i + 1] - bounds[2 * i];
    }
    vtkNew<vtkSignedDistance> distance;
    vtkNew<vtkPCANormalEstimation> normals;

    vtkNew<vtkVertexGlyphFilter> glyphFilter;
    glyphFilter->SetInputData(polyData);
    glyphFilter->Update();

    normals->SetInputData(polyData);
    normals->SetSampleSize(sampleSize);
    normals->SetNormalOrientationToGraphTraversal();
    normals->FlipNormalsOn();
    distance->SetInputConnection(normals->GetOutputPort());

    int dimension = 2048;
    double radius;
    radius = std::max(std::max(range[0], range[1]), range[2]) / static_cast<double>(dimension) * 4; // ~4 voxels

    distance->SetRadius(radius);
    distance->SetDimensions(dimension, dimension, dimension);
    distance->SetBounds(bounds[0] - range[0] * .1, bounds[1] + range[0] * .1,
        bounds[2] - range[1] * .1, bounds[3] + range[1] * .1,
        bounds[4] - range[2] * .1, bounds[5] + range[2] * .1);

    /* Surface Extract
    vtkNew<vtkExtractSurface> surface;
    surface->SetInputConnection(distance->GetOutputPort());
    surface->SetRadius(radius * .99);
    surface->Update();
    */
    /* Poisson Extract
    vtkSmartPointer<vtkPoissonReconstruction> surface = vtkSmartPointer<vtkPoissonReconstruction>::New();
    surface->SetDepth(12);
    surface->SetInputConnection(normals->GetOutputPort());
    */
    /* Crust Extract
    vtkSmartPointer<vtkPowerCrustSurfaceReconstruction> surface = vtkSmartPointer<vtkPowerCrustSurfaceReconstruction>::New();
    surface->SetInputData(polyData);
    */
    // Construct the surfaceand create isosurface.
    vtkNew<vtkSurfaceReconstructionFilter> surf;
    surf->SetInputData(polyData);
    surf->SetNeighborhoodSize(10);
    surf->SetSampleSpacing(0.1);

    vtkNew<vtkContourFilter> contourFilter;
    contourFilter->SetInputConnection(surf->GetOutputPort());
    contourFilter->SetValue(0, 0.0);

    // Sometimes the contouring algorithm can create a volume whose gradient
    // vector and ordering of polygon (using the right hand rule) are
    // inconsistent. vtkReverseSense cures this problem.
    vtkNew<vtkReverseSense> surface;
    surface->SetInputConnection(contourFilter->GetOutputPort());
    surface->ReverseCellsOn();
    surface->ReverseNormalsOn();
    surface->Update();
    // auto newSurf = transform_back(points, reverse->GetOutput());
    //
    vtkNew<vtkPolyDataMapper> map;
    map->SetInputConnection(surface->GetOutputPort());
    map->ScalarVisibilityOff();

    vtkNew<vtkSTLWriter> stlWriter;
    stlWriter->SetFileName("ExtractedSurface.stl");
    stlWriter->SetInputConnection(surface->GetOutputPort());
    stlWriter->Write();
    
    /*
    vtkNew<vtkSTLWriter> stlWriterDiff;
    stlWriterDiff->SetFileName("InitialModel.stl");
    stlWriterDiff->SetInputConnection(diffFilter->GetOutputPort());
    stlWriterDiff->Write();
    */
    vtkNew<vtkPolyDataMapper> surfaceMapper;
    surfaceMapper->SetInputConnection(surface->GetOutputPort());

    vtkNew<vtkProperty> back;
    back->SetColor(colors->GetColor3d("Banana").GetData());

    vtkNew<vtkActor> surfaceActor;
    surfaceActor->SetMapper(surfaceMapper);
    surfaceActor->GetProperty()->SetColor(colors->GetColor3d("Tomato").GetData());
    surfaceActor->SetBackfaceProperty(back);

    renderer->AddActor(surfaceActor);

    renderer->SetBackground(colors->GetColor3d("Gray").GetData());
    renderWindow->Render();
    renderWindowInteractor->Start();

    return EXIT_SUCCESS;
}