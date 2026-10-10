function indices = find_Vmeasure_segments(V, Vmeasure)
    % V is the array of Voltages
    % Vmeasure is the measuring Voltage you are looking for
    % indices will return a cell array, where each cell contains the start
    % and end indices of the segments where V == Vmeasure.

    % Find logical array where voltage matches Vmeasure
    is_Vmeasure = (V == Vmeasure);
    
    % Find the indices of transitions (start or end of Vmeasure segments)
    transitions = diff([0 is_Vmeasure 0]);
    starts = find(transitions == 1);  % Find where segments start
    ends = find(transitions == -1) - 1;  % Find where segments end
    
    % Store start and end indices in a cell array
    indices = cell(length(starts), 1);
    for i = 1:length(starts)
        indices{i} = [starts(i), ends(i)];
    end
end
