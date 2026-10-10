function varargout = DualTrapGUI(varargin)
% DUALTRAPGUI MATLAB code for DualTrapGUI.fig
%      DUALTRAPGUI, by itself, creates a new DUALTRAPGUI or raises the existing
%      singleton*.
%
%      H = DUALTRAPGUI returns the handle to a new DUALTRAPGUI or the handle to
%      the existing singleton*.
%
%      DUALTRAPGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DUALTRAPGUI.M with the given input arguments.
%
%      DUALTRAPGUI('Property','Value',...) creates a new DUALTRAPGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DualTrapGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DualTrapGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DualTrapGUI

% Last Modified by GUIDE v2.5 19-Jan-2016 10:10:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DualTrapGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @DualTrapGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
%% 


% --- Executes just before DualTrapGUI is made visible.
%%
function DualTrapGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DualTrapGUI (see VARARGIN)

% Choose default command line output for DualTrapGUI
handles.output = hObject;
handles.Frupture=[];
handles.FErupture={};
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes DualTrapGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DualTrapGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in LoadData.
function LoadData_Callback(hObject, eventdata, handles)
% hObject    handle to LoadData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% Data Loader
[filename, pathname] = uigetfile('*.*', 'Select Data File');

f=find(filename=='.');
fname2=filename(1:f-1);
fname2=[fname2,'_Voltage.dat'];
vp=load([pathname,fname2]);
handles.Power=vp(1:2:end);
handles.Voltage=vp(2:2:end);
handles.Data = load([pathname,filename]);

handles.trackcheck=get(handles.TwoTracks,'value');



handles.x1=(handles.Data(:,4)+handles.Data(:,2))/2;
handles.dx1=(handles.Data(:,4)-handles.Data(:,2))/2;

handles.x2=(handles.Data(:,5)+handles.Data(:,3))/2;
handles.dx2=(handles.Data(:,5)-handles.Data(:,3))/2;


handles.x1s=handles.Data(:,2);
handles.x2s=handles.Data(:,3);


    
    

handles.time=handles.Data(:,1);
handles.state=handles.Data(:,6);

f=find(handles.state==21);

handles.filename=filename;
handles.pathname=pathname;


if~isempty(f)

df=diff(f);

ftether=find(df~=1);

if ~isempty(ftether)
    

    for i=1:length(ftether+1)

        if i==1
            t1=f(1);
            t2=f(ftether(i));
        else
            t1=f(ftether(i-1)+1);
            t2=f(ftether(i));
        end

        tethers(i,:)=[t1,t2];


    end

     tethers(end+1,:)=[f(ftether(end)+1),f(end)];

    for i= 1:length(tethers(:,1))-1

        tethers(i,3)=tethers(i+1,1)-1;
    
end

else
    tethers(1,1:2)=[f(1),f(2)];    
end

tethers(end,3)=length(handles.time);


handles.tethers=tethers;

end
InitialTetherDisplay;

NumTether=length(tethers(:,1));
handles.NumTether=NumTether;
set(handles.TetherNumber,'String',[num2str(NumTether), ' Tethers']);
handles.ct=1;
TetherDataLoad;

guidata(hObject, handles);
%% 
 
%% 



function kpinput_Callback(hObject, eventdata, handles)
% hObject    handle to kpinput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of kpinput as text
%        str2double(get(hObject,'String')) returns contents of kpinput as a double

handles.kp=str2num(get(handles.kpinput,'String'));

guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function kpinput_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kpinput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in PreviousTether.
function PreviousTether_Callback(hObject, eventdata, handles)
% hObject    handle to PreviousTether (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



if handles.ct>1
    handles.ct=handles.ct-1;
    TetherDataLoad;
    
end


guidata(hObject, handles);


% --- Executes on button press in NextTether.
function NextTether_Callback(hObject, eventdata, handles)
% hObject    handle to NextTether (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% ct=str2num(get(handles.CurrentTether,'String'));

if handles.ct<handles.NumTether
    handles.ct=handles.ct+1;
    TetherDataLoad;
    
end
guidata(hObject, handles);
function CurrentTether_Callback(hObject, eventdata, handles)
% hObject    handle to CurrentTether (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of CurrentTether as text
%        str2double(get(hObject,'String')) returns contents of CurrentTether as a double

ct=str2num(get(handles.CurrentTether,'String'))

if ct<handles.NumTether & ct>0
    
    handles.ct=ct; 
    TetherDataLoad;
    
end
guidata(hObject, handles);
% --- Executes during object creation, after setting all properties.
function CurrentTether_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrentTether (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in GetRupture.
function GetRupture_Callback(hObject, eventdata, handles)
% hObject    handle to GetRupture (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


GetRuptureForce2


guidata(hObject, handles);

function Min_dF_Callback(hObject, eventdata, handles)
% hObject    handle to Min_dF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Min_dF as text
%        str2double(get(hObject,'String')) returns contents of Min_dF as a double


% --- Executes during object creation, after setting all properties.
function Min_dF_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Min_dF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Max_dF_Callback(hObject, eventdata, handles)
% hObject    handle to Max_dF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Max_dF as text
%        str2double(get(hObject,'String')) returns contents of Max_dF as a double


% --- Executes during object creation, after setting all properties.
function Max_dF_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Max_dF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function NumPoints_Callback(hObject, eventdata, handles)
% hObject    handle to NumPoints (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of NumPoints as text
%        str2double(get(hObject,'String')) returns contents of NumPoints as a double


% --- Executes during object creation, after setting all properties.
function NumPoints_CreateFcn(hObject, eventdata, handles)
% hObject    handle to NumPoints (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function dIndex_Callback(hObject, eventdata, handles)
% hObject    handle to dIndex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of dIndex as text
%        str2double(get(hObject,'String')) returns contents of dIndex as a double


% --- Executes during object creation, after setting all properties.
function dIndex_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dIndex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in AcceptRuptureData.
function AcceptRuptureData_Callback(hObject, eventdata, handles)
% hObject    handle to AcceptRuptureData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
FEtransition=handles.FEtransition;
handles.Frupture(end+1,:)=[handles.Ftrans,handles.Ftrans_error,handles.FpS,FEtransition.dz];
i=length(handles.FErupture)+1;

FEtransition.filename=handles.filename;
FEtransition.pathname=handles.pathname;

FEtransition.kp=handles.kp;
FEtransition.tethernumber=handles.ct;

FEtransition.FErun=handles.cFE;
FEtransition.FEall=[handles.z,handles.Ft];

handles.FErupture{i} = FEtransition;

guidata(hObject, handles);
% --- Executes on button press in SaveRuptureData.
function SaveRuptureData_Callback(hObject, eventdata, handles)
% hObject    handle to SaveRuptureData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


assignin('base','FtransTemp',handles.Frupture);
assignin('base','FEtransTemp',handles.FErupture);


% --- Executes on button press in PreviousFE.
function PreviousFE_Callback(hObject, eventdata, handles)
% hObject    handle to PreviousFE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


if handles.cFE>1
    handles.cFE=handles.cFE-1;
   UpdateSubTether; 
    set(handles.CurrentFE,'String',num2str(handles.cFE));
end


guidata(hObject, handles);

% --- Executes on button press in NextFE.
function NextFE_Callback(hObject, eventdata, handles)
% hObject    handle to NextFE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if handles.cFE<length(handles.subtethers(:,1))
   handles.cFE=handles.cFE+1;
   UpdateSubTether; 
   set(handles.CurrentFE,'String',num2str(handles.cFE));
end

guidata(hObject, handles);
function CurrentFE_Callback(hObject, eventdata, handles)
% hObject    handle to CurrentFE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of CurrentFE as text
%        str2double(get(hObject,'String')) returns contents of CurrentFE as a double



% --- Executes during object creation, after setting all properties.
function CurrentFE_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrentFE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function lpg_Callback(hObject, eventdata, handles)
% hObject    handle to lpg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of lpg as text
%        str2double(get(hObject,'String')) returns contents of lpg as a double


% --- Executes during object creation, after setting all properties.
function lpg_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lpg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Lc1g_Callback(hObject, eventdata, handles)
% hObject    handle to Lc1g (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Lc1g as text
%        str2double(get(hObject,'String')) returns contents of Lc1g as a double


% --- Executes during object creation, after setting all properties.
function Lc1g_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Lc1g (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function zstart_Callback(hObject, eventdata, handles)
% hObject    handle to zstart (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of zstart as text
%        str2double(get(hObject,'String')) returns contents of zstart as a double


% --- Executes during object creation, after setting all properties.
function zstart_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zstart (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function zend_Callback(hObject, eventdata, handles)
% hObject    handle to zend (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of zend as text
%        str2double(get(hObject,'String')) returns contents of zend as a double


% --- Executes during object creation, after setting all properties.
function zend_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zend (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in TwoTracks.
function TwoTracks_Callback(hObject, eventdata, handles)
% hObject    handle to TwoTracks (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of TwoTracks
