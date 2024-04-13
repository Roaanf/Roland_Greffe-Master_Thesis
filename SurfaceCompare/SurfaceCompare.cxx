#include <vtkActor.h>
#include <vtkCleanPolyData.h>
#include <vtkDistancePolyDataFilter.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataReader.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkScalarBarActor.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkPolyDataWriter.h>

int main(int argc, char* argv[])
{
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
    targetReader->SetFileName("./Data/Poison14CropFiltered.stl");
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
    vtkWriter->SetFileName("./Data/CompPois14Filtered.vtk");
    vtkWriter->SetInputConnection(distanceFilter->GetOutputPort());
    vtkWriter->Write();

    renWin->Render();
    renWinInteractor->Start();

    return EXIT_SUCCESS;
}