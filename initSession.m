% initSession.m
%
% Run this once at the start of each MATLAB session before using any
% simulation scripts in this project.
%
% Always loads BrainardLabToolbox and Psychtoolbox-3 together so that the
% Staircase class is initialised with its full field set from the outset.
% This avoids the 'clear classes' step that would otherwise be needed when
% switching between 'standard' and 'quest' staircase types.
%
% Usage:
%   >> initSession

clear classes
tbUse({'BrainardLabToolbox', 'Psychtoolbox-3'});
fprintf('Session initialised. Ready to run standard and quest staircases.\n');
