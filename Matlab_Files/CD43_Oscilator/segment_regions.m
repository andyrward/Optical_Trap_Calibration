function [four_regions, three_regions] = segment_regions(arr)
    % Initialize empty arrays to store start and end points for each region
    four_regions = [];
    three_regions = [];

    % Find changes in the values of the array
    changes = [1, find(diff(arr) ~= 0) + 1, length(arr) + 1];

    % Loop through the changes and classify regions
    for i = 1:length(changes) - 1
        region_start = changes(i);
        region_end = changes(i + 1) - 1;
        region_value = arr(region_start);

        if region_value == 4
            four_regions = [four_regions; region_start, region_end];
        elseif region_value == 3
            three_regions = [three_regions; region_start, region_end];
        end
    end
end
