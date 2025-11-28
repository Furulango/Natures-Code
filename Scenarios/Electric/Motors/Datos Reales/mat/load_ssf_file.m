function [data, names, units] = load_ssf_file(filename)
% Open and read file header
fid = fopen(filename);
val = fread(fid,16);
hsize = 0;
for i=1:4
  hsize = hsize*256;
  hsize = hsize+val(5-i);
end
ssfID = char(val(5:8)');
if(!strcmp(ssfID,"SSJH")) % Validate file format
  fprintf("File has not SSJH format\n");
  fclose(fid);
  return;
end
% Load header file
header = char(fread(fid,hsize)');
fclose(fid);
% Decode JSON header file
meta = jsondecode(header);
% Interpret binary data based on the declared header
[data, names, units] = import_bin_with_json(meta,hsize+16,filename);

end
