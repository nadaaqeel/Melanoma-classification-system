% clear all work space and variable 
clear
clc
close all
fontSize = 12 ;
mask_image  = imread(['C:\Users\nadaa\OneDrive\Documents\MATLAB\lesionmasks\ISIC_0000016_Segmentation.png']);

% Orgnize Data 
%----------------------------------------------------------------------------------------
lesionimagesfolder = 'lesionimages/';
lesionmasksfolder  = 'lesionmasks/';
if ~isdir(lesionimagesfolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
lesionPattern = fullfile(lesionimagesfolder, '*.jpg');
imgFiles = dir(lesionPattern);
msksPattern = fullfile(lesionmasksfolder, '*.png');
mskFiles = dir(msksPattern);
for k = 1:length(imgFiles)
  imgFileName = imgFiles(k).name;
  fullFileName = fullfile(lesionimagesfolder, imgFileName);
  imageArray{k} = imread(fullFileName);
  % masks img read 
  mskFileName = mskFiles(k).name;
  fullFileName = fullfile(lesionmasksfolder, mskFileName);
  masksArray{k}= imread(fullFileName);
end

% Load the ground truth table
truth = readtable("ground_truth.csv");

%Maximum index of images
[m n] = size(imageArray);


% Pre- processing Stage (Using Media Filter to smoothen the are and remove small objects, and
% DullRazer Algorithm to remve thik areas
%----------------------------------------------------------------------------------------

% Feature Extraction Stage 
%----------------------------------------------------------------------------------------
% Assymetric feature i extracted based of a Centeroid points of the 
% I have used the provided masks for this operation for their sharp edge
% As i didn't get a good result doing my own segmentation

for i=1:length(masksArray)
 mask_image =  cell2mat(masksArray(i))
 [labeledImage, numberOfObjects] = bwlabel(mask_image);
 measurements = regionprops(labeledImage, 'Centroid');
 m = struct2table(measurements) 
 xCentroids = m.Centroid(:,1)
 yCentroids = m.Centroid(:,2)

 % Find the column and row from the centroid to the image dimentions
 xCentroidColumns = int32(xCentroids)
 yCentroidColumns = int32(yCentroids)

 hold on;
 % Plot centroids
 for k = 1 : numberOfObjects
   plot(xCentroids(k), yCentroids(k), 'ro', 'Markersize', 10, 'linewidth', 1);
 end
 % Find vertical and horizontal lines
for k = 1 : numberOfObjects
  IOR = ismember(labeledImage, k);
  %  top and bottom columns of the blob.
  topRow = find(IOR(:,xCentroidColumns(k)), 1, 'first');
  bottomRow = find(IOR(:,xCentroidColumns(k)), 1, 'last');
  % Horizontal lines
  leftColumn = find(IOR(yCentroidColumns(k), :), 1, 'first');
  rightColumn = find(IOR(yCentroidColumns(k), :), 1, 'last');
  % save Value to use later clockwise
  for x=1:5
  RowColumnCalculation(i,:,:,:,:) = [ topRow , rightColumn , bottomRow , leftColumn ]
  end
end
% find diffrence and normlize 
x_diffrenceArray(i) = abs(topRow - bottomRow) / (  bottomRow + topRow) 
y_diffrenceArray(i) = abs(rightColumn - leftColumn) / (  bottomRow + topRow) 
% I have chosen the y array values as many of the lesion images shows a
% assymetrical vertical shape more obvoius than the horizontal one
Assyme_value(i) =  y_diffrenceArray(i)
fearures_values(i,1) = Assyme_value(i) 
end

% Feature Extraction Stage 
%----------------------------------------------------------------------------------------
% Colour pallet , K means is applied to the masked rgb images , 
% to extract the color 4 pallet for the images and Euclidian Distance is
% applid

% First Crop the inhanced images using provided masks to only get the area
% of the lesion colours
% estract region of intrest using masks

for i= 1:length(imageArray)
 mask_image =  cell2mat(masksArray(i))
 image = cell2mat(imageArray(i))
 mask_image(:,:,2) = mask_image
 mask_image(:,:,3) = mask_image(:,:,1)
 IRO = image
 IRO(mask_image == 0) = 0
% 6) Display extracted portion:
%end
% crop the image to the selected lesion area
   topRow = RowColumnCalculation(i, 1  ) 
   rightColumn =RowColumnCalculation(i,2) 
   bottomRow = RowColumnCalculation(i, 3) 
   leftColumn = RowColumnCalculation(i, 4) 

% Crop image with some added padding to confirm shap
lesionarea = imcrop(IRO, [leftColumn - 50  , topRow , (rightColumn - leftColumn) + 100 , (bottomRow - topRow) + 20]);
lsionlist{i} = lesionarea

end


% arrayOfFeatures = colorPallet(lsionlist)

% Feature Extraction Stage 
%----------------------------------------------------------------------------------------
% Colour pallet , K means is applied to the masked rgb images , 
% to extract the color 4 pallet for the images and Euclidian Distance is
% applid
% Get the dimensions of the image.  numberOfColorChannels should be = 3.

% function fearures_values = colorPallet(lesionarealist)

for i=1:length(lsionlist)
    
 lesionarea =  cell2mat(lsionlist(i))
 [rows, columns, numberOfColorChannels] = size(lesionarea);
 % Enlarge figure to full screen.
 set(gcf, 'Units', 'Normalized', 'Outerposition', [0, 0, 1, 1], 'Name', 'Color Channels');

% Extract the individual red, green, and blue color channels.
redChannel = lesionarea(:, :, 1);
greenChannel = lesionarea(:, :, 2);
blueChannel = lesionarea(:, :, 3);

%----------------------------------------------------------------------------------------
% Get the data for doing kmeans.  We will have 3 columns, each with one color channel.
data = double([redChannel(:), greenChannel(:), blueChannel(:)]);
%  kmeans decide which cluster each pixel belongs to.
numberOfClasses = 5;
indexes = kmeans(data, numberOfClasses);

%----------------------------------------------------------------------------------------
% Let's convert what class index the pixel is into images for each class index.
class1 = reshape(indexes == 1, rows, columns);
class2 = reshape(indexes == 2, rows, columns);
class3 = reshape(indexes == 3, rows, columns);
class4 = reshape(indexes == 4, rows, columns);
class5 = reshape(indexes == 5, rows, columns);
class6 = reshape(indexes == 6, rows, columns);
% to 3-D array for later to make it easy to display them all with a loop.
allClasses = cat(3, class1, class2, class3, class4, class5, class6);
allClasses = allClasses(:, :, 1:numberOfClasses); % Crop off just what we need.
%----------------------------------------------------------------------------------------
% Compute an indexed image for comparison;
[indexedImage, customColorMap]  = rgb2ind(lesionarea, numberOfClasses);

% subplot(3, numberOfClasses, 2);
h3 = subplot(1, 1, 1);
colormap(h3, customColorMap);

% calculate the distace between each 2 pixels from tha color map
% start at 0 to include the black from 1
for i= 2:length(customColorMap )
 dist_E(i) = sqrt((customColorMap(i) - customColorMap(i+1)).^2 + (customColorMap(i) - customColorMap(i+1)).^2)
end
% normlize
dist_E = sum(dist_E) /4 
% Features values 
fearures_values(i,2) = dist_E

 end
% end



% Feature Extraction Stage 
%----------------------------------------------------------------------------------------
% Image Classification Building Classifier
%----------------------------------------------------------------------------------------
% T = readtable('ground_truth.csv');
% test = T.benign
 %svm = fitcsvm([Assyme_value distance_values], T.benign)
% cvsvm = crossval(svm);
%pred = kfoldPredict(cvsvm)
%cm = confusionmat(pred, T.benign)

