% initSession.m
%
% Run this once at the start of each MATLAB session before using any
% simulation scripts in this project.  Loads BrainardLabToolbox and
% Psychtoolbox-3 (required for the 'quest' staircaseType).
%
% Note: if you switch between 'standard' and 'quest' staircase types in
% the same session you may still need to run 'clear classes' manually.
%
% Usage:
%   >> initSession

clear classes
tbUse({'BrainardLabToolbox', 'Psychtoolbox-3'});
fprintf('Session initialised.\n');
