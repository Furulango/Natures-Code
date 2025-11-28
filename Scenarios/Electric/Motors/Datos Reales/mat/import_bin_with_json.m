function [data, names, units] = import_bin_with_json(meta, offset, bin_file)
  % IMPORT_BIN_WITH_JSON  Import a binary file using a JSON definition
  %
  %  data = import_bin_with_json(<JSON structure>, <file offset>, <binary file name>)

  % Import channel number
  nstreams = numel(meta.stream);

  % Open binary file
  fid = fopen(bin_file, "rb");
  if fid < 0
      error("Unable to open file %s", bin_file);
  end
  if(offset!=0)
    fread(fid,offset);

  % Set datatype sizes
  type_sizes = containers.Map( ...
      {'i16','u16','i32','u32','f32','f64'}, ...
      {2,2,4,4,4,8} );

  % Calculate single record size
  record_size = 0;
  for k=1:nstreams
     record_size = record_size + type_sizes(meta.stream(k).type);
  end

  % Read raw data
  raw = fread(fid, Inf, "uint8=>uint8");
  fclose(fid);

  filesize = numel(raw);
  nrecords = floor(filesize / record_size);

  if nrecords == 0
    warning("Empty file or wrong definition");
    data = struct();
    return;
  end

  % Reshape binary data [bytes_per_record × nrecords]
  raw = reshape(raw(1:nrecords*record_size), record_size, nrecords);

  % Apply offset and scale correction
  data = struct();
  offset = 1;
  names = [];
  units = [];
  data.sampling = meta.sampling;

  for k=1:nstreams
      name  = meta.stream(k).name;
      tname = meta.stream(k).type;
      gain  = meta.stream(k).gain;
      offs  = meta.stream(k).offset;
      bsize = type_sizes(tname);

      % Store data names per channel
      names{k} = name;
      units{k} = meta.stream(k).unit;

      % Bytes per record
      bytes_k = raw(offset:offset+bsize-1, :);

      % Datatype conversion
      vec = typecast(reshape(bytes_k, 1, []), map_type(tname));
      vec = reshape(vec, nrecords, 1);

      % Set offset and gain
      data.(name) = double(vec) * gain + offs;
      offset += bsize;
  end
end

