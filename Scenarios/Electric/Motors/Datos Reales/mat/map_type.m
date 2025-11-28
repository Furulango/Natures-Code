% --- Función auxiliar para mapear a tipo Octave/MATLAB ---
function t = map_type(str)
  switch str
    case "i16"
      t = "int16";
    case "u16"
      t = "uint16";
    case "i32"
      t = "int32";
    case "u32"
      t = "uint32";
    case "f32"
      t = "single";
    case "f64"
      t = "double";
    otherwise
      error("Tipo no soportado: %s", str);
  end
end
