% initSession.m
%
% Run this once at the start of each MATLAB session before using any
% simulation scripts in this project.
%
% BrainardLabToolbox's Staircase is an old-style MATLAB class whose field
% layout is fixed by whichever type is instantiated first.  The 'quest'
% type adds a QuestObj field that 'standard' does not.  If 'standard' is
% created first, switching to 'quest' later raises:
%   "Cannot change the number of fields of class 'Staircase'"
%
% Fix: prime the class with a dummy 'quest' Staircase immediately after
% loading both toolboxes.  All subsequent Staircases (standard or quest)
% then share the same full field layout.
%
% Usage:
%   >> initSession

clear classes
tbUse({'BrainardLabToolbox', 'Psychtoolbox-3'});

% Prime Staircase class with the quest field layout.
dummy = Staircase('quest', 1.0, ...
    'Beta', 3.5, 'Delta', 0.01, 'Gamma', 0.0, ...
    'TargetThreshold', 0.75, 'PriorSD', 10, ...
    'MaxValue', 3.0, 'MinValue', 0.01);
clear dummy;

fprintf('Session initialised. Ready to run standard and quest staircases.\n');
