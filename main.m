% clear; clc; close all;
%% Parameter input
Widx = 1; % Optins- Widx = 1/2/3/4 

%% Read input image 
[InputImage] = imread('InputImage/cartoon.jpg');
figure;
imshow(InputImage);
% InputImage = imnoise(InputImage,'gaussian',0,0.05);
InputImage = im2double(InputImage);


%% Call the SGI generation
[FilteredImage] = EdgeMapGeneration(InputImage,Widx); 

%% Show output filtered image
figure;
imshow(FilteredImage);
