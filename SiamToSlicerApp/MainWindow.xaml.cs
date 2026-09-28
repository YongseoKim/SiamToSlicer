using System.Diagnostics;
using System.IO;
using System.Windows;

namespace SiamToSlicerApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private static readonly string OutputDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "SiamizeApp",
            "out");

        private string _volumePath = string.Empty;
        private string _predictionPath = string.Empty;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void DicomButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFolderDialog
            {
                Title = "Choose DICOM folder",
                InitialDirectory = InitialDicomDirectory()
            };

            if (dialog.ShowDialog() == true)
            {
                DicomPathTextBox.Text = dialog.FolderName;
            }
        }

        private async void InferenceButton_Click(object sender, RoutedEventArgs e)
        {
            var dicomFolder = DicomPathTextBox.Text.Trim();
            if (dicomFolder.Length == 0 || !Directory.Exists(dicomFolder))
            {
                AppendLog("Choose DICOM folder first.");
                return;
            }

            DicomButton.IsEnabled = false;
            InferenceButton.IsEnabled = false;
            OpenSlicerButton.IsEnabled = false;
            OpenVolumeButton.IsEnabled = false;
            LogTextBox.Clear();
            var inferenceSucceeded = false;

            try
            {
                var dcm2niix = FindBundledTool("dcm2niix", "dcm2niix.exe");
                if (dcm2niix == null)
                {
                    AppendLog("Failed to find dcm2niix.exe.");
                    return;
                }

                var siamize = FindBundledTool("siamize", "siamize.exe");
                if (siamize == null)
                {
                    AppendLog("Failed to find siamize.exe.");
                    return;
                }

                var model = Path.Combine(Path.GetDirectoryName(siamize)!, "fold_0_fp16.onnx");
                if (!File.Exists(model))
                {
                    AppendLog("Failed to find fold_0_fp16.onnx next to siamize.exe.");
                    return;
                }

                var cudaBin = FindCudaBin();
                if (cudaBin == null)
                {
                    AppendLog("Failed to find CUDA Toolkit 12. Install CUDA 12.9 and try again.");
                    return;
                }

                var cudnnBin = FindCudnnBin(cudaBin);
                if (cudnnBin == null)
                {
                    AppendLog("Failed to find cuDNN 9. Install it for CUDA 12 and try again.");
                    return;
                }

                AppendLog("CUDA: " + cudaBin);
                AppendLog("cuDNN: " + cudnnBin);

                Directory.CreateDirectory(OutputDirectory);
                ClearPreviousVolumes();
                AppendLog("Convert DICOM to NIfTI.");
                var convertExit = await RunProcessAsync(dcm2niix, new[]
                {
                    "-z", "y",
                    "-o", OutputDirectory,
                    "-f", "volume",
                    dicomFolder
                });
                if (convertExit != 0)
                {
                    AppendLog("Failed to Convert. End the code " + convertExit);
                    return;
                }

                var niftiPath = FindConvertedNifti();
                if (niftiPath == null)
                {
                    AppendLog("Failed to find converted NIfTI file.");
                    return;
                }

                var predictionPath = Path.Combine(OutputDirectory, "pred.nii.gz");
                AppendLog("Start the Inference: " + niftiPath);
                var inferExit = await RunProcessAsync(
                    siamize,
                    new[]
                    {
                        "--input", niftiPath,
                        "--output", predictionPath,
                        "-M", model,
                        "-c", "cuda",
                        "-v"
                    },
                    cudaBin,
                    cudnnBin);
                if (inferExit != 0 || !File.Exists(predictionPath))
                {
                    AppendLog("Failed to Convert. End the code " + inferExit);
                    return;
                }

                _volumePath = niftiPath;
                _predictionPath = predictionPath;
                AppendLog("Complete: " + predictionPath);
                inferenceSucceeded = true;
            }
            catch (Exception ex)
            {
                AppendLog(ex.Message);
            }
            finally
            {
                DicomButton.IsEnabled = true;
                InferenceButton.IsEnabled = true;
                OpenSlicerButton.IsEnabled = inferenceSucceeded;
                OpenVolumeButton.IsEnabled = inferenceSucceeded;
            }
        }

        private static string? FindBundledTool(string folderName, string fileName)
        {
            var candidate = Path.Combine(AppContext.BaseDirectory, "runtime", folderName, fileName);
            return File.Exists(candidate) ? candidate : null;
        }

        private static void ClearPreviousVolumes()
        {
            if (!Directory.Exists(OutputDirectory))
            {
                return;
            }

            foreach (var path in Directory.GetFiles(OutputDirectory, "volume*"))
            {
                var name = Path.GetFileName(path);
                if (name.EndsWith(".nii", StringComparison.OrdinalIgnoreCase)
                    || name.EndsWith(".nii.gz", StringComparison.OrdinalIgnoreCase)
                    || name.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                {
                    File.Delete(path);
                }
            }
        }

        private static string? FindConvertedNifti()
        {
            return Directory.GetFiles(OutputDirectory)
                .Where(path =>
                    !string.Equals(Path.GetFileName(path), "pred.nii.gz", StringComparison.OrdinalIgnoreCase)
                    && (path.EndsWith(".nii", StringComparison.OrdinalIgnoreCase)
                        || path.EndsWith(".nii.gz", StringComparison.OrdinalIgnoreCase)))
                .OrderByDescending(File.GetLastWriteTimeUtc)
                .FirstOrDefault();
        }

        private static string? FindCudaBin()
        {
            var fromEnvironment = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (HasCuda12Runtime(fromEnvironment))
            {
                return Path.Combine(fromEnvironment!, "bin");
            }

            var root = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "NVIDIA GPU Computing Toolkit",
                "CUDA");
            if (!Directory.Exists(root))
            {
                return null;
            }

            return Directory.EnumerateDirectories(root, "v12*")
                .Select(dir => new { Dir = dir, Version = ParseVersion(Path.GetFileName(dir)) })
                .Where(item => item.Version != null && HasCuda12Runtime(item.Dir))
                .OrderByDescending(item => item.Version)
                .Select(item => Path.Combine(item.Dir, "bin"))
                .FirstOrDefault();
        }

        private static string? FindCudnnBin(string cudaBin)
        {
            if (File.Exists(Path.Combine(cudaBin, "cudnn64_9.dll")))
            {
                return cudaBin;
            }

            var cudaVersion = ParseVersion(Path.GetFileName(Path.GetDirectoryName(cudaBin)));
            var cudnnRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "NVIDIA",
                "CUDNN");
            if (!Directory.Exists(cudnnRoot))
            {
                return null;
            }

            var matches = Directory.EnumerateFiles(cudnnRoot, "cudnn64_9.dll", SearchOption.AllDirectories)
                .Select(Path.GetDirectoryName)
                .OfType<string>()
                .Where(dir => dir.Contains($"{Path.DirectorySeparatorChar}12.", StringComparison.OrdinalIgnoreCase))
                .Select(dir => new { Dir = dir, Version = CudaVersionFromCudnnPath(dir) })
                .Where(item => item.Version != null && item.Version.Major == 12)
                .OrderByDescending(item => cudaVersion != null && item.Version == cudaVersion)
                .ThenByDescending(item => item.Version)
                .Select(item => item.Dir)
                .FirstOrDefault();
            return matches;
        }

        private static bool HasCuda12Runtime(string? cudaRoot)
        {
            return !string.IsNullOrWhiteSpace(cudaRoot)
                && File.Exists(Path.Combine(cudaRoot, "bin", "cudart64_12.dll"));
        }

        private static Version? ParseVersion(string? name)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                return null;
            }

            var text = name.StartsWith("v", StringComparison.OrdinalIgnoreCase) ? name[1..] : name;
            return Version.TryParse(text, out var version) ? version : null;
        }

        private static Version? CudaVersionFromCudnnPath(string directory)
        {
            var parts = directory.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            for (var i = 0; i < parts.Length; ++i)
            {
                if (parts[i].Equals("bin", StringComparison.OrdinalIgnoreCase) && i + 1 < parts.Length)
                {
                    return ParseVersion(parts[i + 1]);
                }
            }

            return null;
        }

        private async Task<int> RunProcessAsync(
            string fileName,
            IReadOnlyList<string> arguments,
            string? cudaBin = null,
            string? cudnnBin = null)
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = fileName,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = Path.GetDirectoryName(fileName) ?? Environment.CurrentDirectory
            };

            foreach (var argument in arguments)
            {
                startInfo.ArgumentList.Add(argument);
            }

            if (cudaBin != null)
            {
                var currentPath = startInfo.Environment["PATH"];
                var prefix = cudaBin;
                if (cudnnBin != null
                    && !string.Equals(cudnnBin, cudaBin, StringComparison.OrdinalIgnoreCase))
                {
                    prefix = cudnnBin + Path.PathSeparator + prefix;
                }

                startInfo.Environment["PATH"] = prefix + Path.PathSeparator + currentPath;
            }

            using var process = new Process { StartInfo = startInfo };
            process.OutputDataReceived += (_, args) => AppendLog(args.Data);
            process.ErrorDataReceived += (_, args) => AppendLog(args.Data);
            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();
            return process.ExitCode;
        }

        private void OpenSlicerButton_Click(object sender, RoutedEventArgs e)
        {
            var slicerPath = FindSlicer();
            if (slicerPath == null)
            {
                MessageBox.Show(
                    "Failed to find 3D Slicer. Download the program.",
                    "3D Slicer",
                    MessageBoxButton.OK,
                    MessageBoxImage.Warning);
                return;
            }

            var python = $$"""
                import qt
                import slicer
                slicer.util.loadVolume(r'{{_volumePath}}')
                slicer.util.loadLabelVolume(r'{{_predictionPath}}')
                def showSliceViewers():
                    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
                    slicer.util.resetSliceViews()
                qt.QTimer.singleShot(0, showSliceViewers)
                """;

            StartSlicer(slicerPath, python);
        }

        private void OpenVolumeButton_Click(object sender, RoutedEventArgs e)
        {
            var slicerPath = FindSlicer();
            if (slicerPath == null)
            {
                MessageBox.Show(
                    "Failed to find 3D Slicer. Download the program.",
                    "3D Slicer",
                    MessageBoxButton.OK,
                    MessageBoxImage.Warning);
                return;
            }

            var python = $$"""
                import qt
                import slicer
                volumeNode = slicer.util.loadVolume(r'{{_volumePath}}')
                slicer.util.loadLabelVolume(r'{{_predictionPath}}')
                volRenLogic = slicer.modules.volumerendering.logic()
                displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
                displayNode.SetVisibility(True)
                scalarRange = volumeNode.GetImageData().GetScalarRange()
                presetName = 'MR-Default' if scalarRange[1] - scalarRange[0] < 1500 else 'CT-Chest-Contrast-Enhanced'
                displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName(presetName))
                def showVolumeRendering():
                    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
                    slicer.util.resetThreeDViews()
                qt.QTimer.singleShot(0, showVolumeRendering)
                """;

            StartSlicer(slicerPath, python);
        }

        private static void StartSlicer(string slicerPath, string python)
        {
            var scriptPath = Path.Combine(Path.GetTempPath(), "SiamToSlicer-" + Guid.NewGuid().ToString("N") + ".py");
            File.WriteAllText(scriptPath, python);
            Process.Start(new ProcessStartInfo
            {
                FileName = slicerPath,
                UseShellExecute = false,
                ArgumentList = { "--python-script", scriptPath }
            });
        }

        private static string? FindSlicer()
        {
            var found = new List<string>();
            var localRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "slicer.org");
            if (Directory.Exists(localRoot))
            {
                found.AddRange(Directory.EnumerateFiles(localRoot, "Slicer.exe", SearchOption.AllDirectories));
            }

            foreach (var programFiles in new[]
            {
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86)
            })
            {
                if (!Directory.Exists(programFiles))
                {
                    continue;
                }

                foreach (var dir in Directory.EnumerateDirectories(programFiles, "*Slicer*"))
                {
                    var exe = Path.Combine(dir, "Slicer.exe");
                    if (File.Exists(exe))
                    {
                        found.Add(exe);
                    }
                }
            }

            return found
                .OrderByDescending(File.GetLastWriteTimeUtc)
                .FirstOrDefault();
        }

        private static string InitialDicomDirectory()
        {
            var downloads = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "Downloads");
            return Directory.Exists(downloads)
                ? downloads
                : Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        }

        private void AppendLog(string? line)
        {
            if (string.IsNullOrEmpty(line))
            {
                return;
            }

            Dispatcher.Invoke(() =>
            {
                LogTextBox.AppendText(line + Environment.NewLine);
                LogTextBox.ScrollToEnd();
            });
        }
    }
}
