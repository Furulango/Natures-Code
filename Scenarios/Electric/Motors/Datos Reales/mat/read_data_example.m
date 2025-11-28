
close all
clear all

% Indicar la ruta de la carpeta y el nombre del archivo a visualizar
folder_path = "./";
file_name = "elec0000.ssf";
file = strcat(folder_path,'/',file_name);
[data, names, units] = load_ssf_file(file);

% Frecuencia de muestreo
fs = data.sampling;

% Para hacer referencia a un canal de datos en particular puede acceder mediante data.<nombre>, e.g.: data.ax
t = (0:length(data.(names{1}))-1)/fs;

% Genera la gráfica de los canales de datos
figure(1)
vars = {};
k=1;
for i=1:length(names)
  vars{k} = names{i};
  varu{k} = units{i};
  y = data.(names{i});
  k = k + 1;
  plot(t,y)
  hold on
end
xlabel("Tiempo (s)");
ylabel("Amplitud")
title("Gráfica de los datos almacenados")
legend(strcat(vars," (",varu,")"))
set(gca,'fontsize',12)
