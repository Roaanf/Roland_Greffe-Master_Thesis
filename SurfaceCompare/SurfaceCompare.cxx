#include <vtkActor.h>
#include <vtkCleanPolyData.h>
#include <vtkDistancePolyDataFilter.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkCylinder.h>
#include <vtkPolyDataReader.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkScalarBarActor.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkSTLReader.h>
#include <vtkConvertToPointCloud.h>
#include <vtkSTLWriter.h>
#include <vtkImplicitBoolean.h>
#include <vtkPolyDataWriter.h>
#include <vtkExtractPolyDataGeometry.h>
#include <vtkStaticPointLocator.h>
#include <vtkMath.h>
#include <vtkVector.h>

int main(int argc, char* argv[])
{
    bool computePreciseErrors = true;
    std::string userAnswer;
    std::cout << "Compute precise errors ? (0/1)" << std::endl;
    std::cin >> userAnswer;
    if (userAnswer == "0" || userAnswer == "n" || userAnswer == "N") {
		computePreciseErrors = false;
	}

    vtkNew<vtkNamedColors> colors;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);
    renderWindow->SetWindowName("SurfaceCompare");
    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Load Reference STL
    vtkNew<vtkSTLReader> referenceReader;
    referenceReader->SetFileName("./Data/InitialModel.stl");
    referenceReader->Update();

    vtkNew<vtkCleanPolyData> referencePolyData;
    referencePolyData->SetInputData(referenceReader->GetOutput());

    // Load Target STL
    vtkNew<vtkSTLReader> targetReader;
    targetReader->SetFileName("./Data/SurfaceNets.stl");
    targetReader->Update();

    if (!computePreciseErrors) {

        vtkNew<vtkCleanPolyData> targetPolyData;
        targetPolyData->SetInputData(targetReader->GetOutput());

        vtkNew<vtkDistancePolyDataFilter> distanceFilter;
        distanceFilter->SetInputConnection(1, referencePolyData->GetOutputPort());
        distanceFilter->SetInputConnection(0, targetPolyData->GetOutputPort());
        distanceFilter->SetSignedDistance(false);
        distanceFilter->Update();

        vtkNew<vtkPolyDataMapper> referenceMapper;
        referenceMapper->SetInputConnection(distanceFilter->GetOutputPort());
        referenceMapper->SetScalarRange(
            distanceFilter->GetOutput()->GetPointData()->GetScalars()->GetRange()[0],
            distanceFilter->GetOutput()->GetPointData()->GetScalars()->GetRange()[1]);
        vtkNew<vtkActor> referenceActor;
        referenceActor->SetMapper(referenceMapper);

        vtkNew<vtkScalarBarActor> scalarBar;
        scalarBar->SetLookupTable(referenceMapper->GetLookupTable());
        scalarBar->SetTitle("Distance");
        scalarBar->SetNumberOfLabels(4);
        scalarBar->UnconstrainedFontSizeOn();

        vtkNew<vtkRenderWindow> renWin;
        renWin->AddRenderer(renderer);
        renWin->SetWindowName("DistancePolyDataFilter");

        vtkNew<vtkRenderWindowInteractor> renWinInteractor;
        renWinInteractor->SetRenderWindow(renWin);

        renderer->AddActor(referenceActor);
        renderer->AddActor2D(scalarBar);

        // Doesn't seem to be extractable from the filter ?
        // Will try a .vtk file instead
        vtkNew<vtkPolyDataWriter> vtkWriter;
        vtkWriter->SetFileName("./Data/CompPoison14CropFiltered.vtk");
        vtkWriter->SetInputConnection(distanceFilter->GetOutputPort());
        vtkWriter->Write();
        renWin->Render();
        renWinInteractor->Start();
    
    } else {
        // Length of edges
        vtkNew<vtkStaticPointLocator> pointLocator;
        vtkPolyData* polyData = targetReader->GetOutput();
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
        for (size_t i = 0; i < 16; i++) {
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
            }
            else {
                distancesError[i] = abs(distance - 41.2309);
            }
            std::cout << distancesError[i] << " ";
        }
        std::cout << std::endl;
        // Angles
        double anglesErrors[3]; // Voir si j'en fais plus ?
        double realAngles[3] = { 90.0, 90.0, 43.313};
        double point1Vec[4][3] = { {20.0, -20.0, -20.0},{-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},
									  {-20.0, 20.0, 20.0}};
        double point2Vec[4][3] = { {-20.0, -20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, 20.0, 20.0},
									  {0.0, 50.0, 0.0}};
        // Flemme de la boucle polalalalala
        std::vector<vtkVector3d> vectors;
        for (size_t i = 0; i < 4; i++){
            vtkIdType point1ID = pointLocator->FindClosestPoint(point1Vec[i]);
            double* point1Point = polyData->GetPoint(point1ID);
            double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
            vtkIdType point2ID = pointLocator->FindClosestPoint(point2Vec[i]);
            double* point2Point = polyData->GetPoint(point2ID);
            double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
            double diff[3] = { point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2] };
            //create vector
            vtkVector3d vector = vtkVector3d(diff);
            vectors.push_back(vector);
        }

        anglesErrors[0] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[0].GetData(), vectors[2].GetData())) - realAngles[0];
        anglesErrors[1] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[1].GetData(), vectors[2].GetData())) - realAngles[1];
        anglesErrors[2] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[3].GetData(), vectors[2].GetData())) - realAngles[2];

        std::cout << anglesErrors[0] << " " << anglesErrors[1] << " " << anglesErrors[2] << std::endl;
        vtkNew<vtkCylinder> cylinder;
        cylinder->SetCenter(-20.0, 0.0, 0.0);
        cylinder->SetRadius(10.15);
        cylinder->SetAxis(1.0, 0.0, 0.0);
        vtkNew<vtkImplicitBoolean> boolean;
        boolean->AddFunction(cylinder);
        vtkNew<vtkExtractPolyDataGeometry> extractPolyDataGeometry;
        extractPolyDataGeometry->SetInputData(polyData);
        extractPolyDataGeometry->SetExtractInside(true);
        extractPolyDataGeometry->SetImplicitFunction(boolean);
        extractPolyDataGeometry->Update();

        vtkNew<vtkConvertToPointCloud> pcConvert;
        pcConvert->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
        pcConvert->SetCellGenerationMode(vtkConvertToPointCloud::NO_CELLS);
        pcConvert->Update();
        vtkPolyData* cylPoints = pcConvert->GetOutput();
        std::ofstream myfile;
        myfile.open("PointCloud.txt");
        for (size_t i = 0; i < cylPoints->GetNumberOfPoints(); i++) {
            double* point = cylPoints->GetPoint(i);
            myfile << point[0] << " " << point[1] << " " << point[2] << "\n";
        }
        myfile.close();
        vtkNew<vtkPolyDataWriter> vtkWriter;
        vtkWriter->SetFileName("./Data/TestCylinderCrop.vtk");
        vtkWriter->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
        vtkWriter->Write();
        
        // Do the fitting -> https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf (banger)
        // Flemme de le coder je vais compiler GeometricTools à la place UwU
        bool feur;
        std::cin >> feur;

    }

    return EXIT_SUCCESS;
}