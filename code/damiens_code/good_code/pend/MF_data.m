
clear; clc;
close all

X1 = linspace(0, 1.0, 44); 
Xout = linspace(0, 1.0, 101); 

rng(123)
A = linspace(14, 20, 100); 
counter = 0; 
for acur = A
        b1 = acur*X1 -4;
        bout = acur*Xout -4;
        y1 = X1+sin(b1); 
        ydiffout = Xout+sin(bout);

        
        
        %SAVE LF
        Input_branch = b1;
        Target = ydiffout;
        Input_trunk1 = Xout; 
        saveFolder = ['LFdata_full/', 'beta_', num2str(counter), '_'];
        saveName = ['data_train.mat'];
        save([saveFolder, saveName],'Input_branch','Input_trunk1', 'Target');

        
        %SAVE LF
        Input_branch = b1;
        Target = y1;
        Input_trunk1 = X1; 
        saveFolder = ['LFdata/', 'beta_', num2str(counter), '_'];
        saveName = ['data_train.mat'];
        save([saveFolder, saveName],'Input_branch','Input_trunk1', 'Target');
        
        
        
        
        
        counter  = counter + 1;
        plot(X1, y1)
        hold on
end

counter = 0;


A = linspace(14.1, 19.9, 20); 
figure

for acur = A
        b1 = acur*X1 -4;
        bout = acur*Xout -4;
        y1 = X1+sin(b1); 
        ydiffout = Xout+sin(bout);

        
        
                %SAVE LF
        Input_branch = b1;
        Target = ydiffout;
        Input_trunk1 = Xout; 
        saveFolder = ['LFdata_full/', 'beta_', num2str(counter), '_'];
        saveName = ['data_test.mat'];
        save([saveFolder, saveName],'Input_branch','Input_trunk1', 'Target');

        
        %SAVE LF
        Input_branch = b1;
        Target = y1;
        Input_trunk1 = X1; 
        saveFolder = ['LFdata/', 'beta_', num2str(counter), '_'];
        saveName = ['data_test.mat'];
        save([saveFolder, saveName],'Input_branch','Input_trunk1', 'Target');
        
   
        counter  = counter + 1;
        plot(X1, y1)
        hold on
end





