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
#include <vtkImageData.h>
#include <vtkImageStencil.h>
#include <vtkMetaImageWriter.h>
#include <vtkConvertToPointCloud.h>
#include <vtkPolyDataToImageStencil.h>
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

    //Generate a voxel volume from the reference model
    bool genereateRefVoxelData = true;
    std::cout << "Generate MHD of ref model ? (0/1)" << std::endl;
    std::cin >> userAnswer;
    if (userAnswer == "0" || userAnswer == "n" || userAnswer == "N") {
        genereateRefVoxelData = false;
    }

    if (genereateRefVoxelData) {
        // Stuff to generate the voxel volume
        vtkNew<vtkImageData> whiteImage;
        double bounds[6];
        referenceReader->GetOutput()->GetBounds(bounds);
        double spacing[3]; // desired volume spacing
        spacing[0] = 0.2; // This spacing leads to a volume close to the original reconstructed volume (200*350*200)
        spacing[1] = 0.2;
        spacing[2] = 0.2;
        whiteImage->SetSpacing(spacing);

        // compute dimensions
        int dim[3];
        for (int i = 0; i < 3; i++)
        {
            dim[i] = static_cast<int>(
                ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i]));
        }
        whiteImage->SetDimensions(dim);
        whiteImage->SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1);
        
        double origin[3];
        origin[0] = bounds[0] + spacing[0] / 2;
        origin[1] = bounds[2] + spacing[1] / 2;
        origin[2] = bounds[4] + spacing[2] / 2;
        whiteImage->SetOrigin(origin);
        whiteImage->AllocateScalars(VTK_UNSIGNED_SHORT, 1);

        // Fill the image with foreground voxels:
        unsigned char inval = 255;
        unsigned char outval = 0;
        vtkIdType count = whiteImage->GetNumberOfPoints();
        for (vtkIdType i = 0; i < count; ++i)
        {
            whiteImage->GetPointData()->GetScalars()->SetTuple1(i, inval);
        }

        vtkNew<vtkPolyDataToImageStencil> pol2stenc;
        pol2stenc->SetInputData(referenceReader->GetOutput());
        pol2stenc->SetOutputOrigin(origin);
        pol2stenc->SetOutputSpacing(spacing);
        pol2stenc->SetOutputWholeExtent(whiteImage->GetExtent());
        pol2stenc->Update();

        vtkNew<vtkImageStencil> imgstenc;
        imgstenc->SetInputData(whiteImage);
        imgstenc->SetStencilConnection(pol2stenc->GetOutputPort());
        imgstenc->ReverseStencilOff();
        imgstenc->SetBackgroundValue(outval);
        imgstenc->Update();

        vtkNew<vtkMetaImageWriter> writer;
        writer->SetFileName("ReferenceVoxelWONoise.mhd");
        writer->SetInputData(imgstenc->GetOutput());
        writer->Write();
        return EXIT_SUCCESS;

	}

    vtkNew<vtkCleanPolyData> referencePolyData;
    referencePolyData->SetInputData(referenceReader->GetOutput());

    // Load Target STL
    vtkNew<vtkSTLReader> targetReader;
    targetReader->SetFileName("./Data/PoissonNoNoise.stl");
    targetReader->Update();


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
    vtkWriter->SetFileName("./Data/PoissonNoNoise.vtk");
    vtkWriter->SetInputConnection(distanceFilter->GetOutputPort());
    vtkWriter->Write();
    renWin->Render();
    renWinInteractor->Start();
    
    if (computePreciseErrors) {
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