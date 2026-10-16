import { MyComposition } from "./Composition";
import { Composition, Folder } from "remotion";
import { HazeClearExplainer } from "./hazeclear/HazeClearExplainer";
import { Opening } from "./hazeclear/Opening";
import { Wavelet } from "./hazeclear/Wavelet";
import { Mamba } from "./hazeclear/Mamba";
import { Reconstruction } from "./hazeclear/Reconstruction";
import { DenRole } from "./hazeclear/DenRole";
import { Detail } from "./hazeclear/Detail";
import { DenResult } from "./hazeclear/DenResult";
import { Optimization } from "./hazeclear/Optimization";
import { Closing } from "./hazeclear/Closing";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="HazeClearExplainer"
        component={HazeClearExplainer}
        durationInFrames={2700}
        fps={30}
        width={1280}
        height={720}
      />
      <Folder name="HazeClear-Scenes">
        <Composition
          id="HazeClear-Opening"
          component={Opening}
          durationInFrames={240}
          fps={30}
          width={1280}
          height={720}
        />
        <Composition
          id="HazeClear-Wavelet"
          component={Wavelet}
          durationInFrames={300}
          fps={30}
          width={1280}
          height={720}
        />
        <Composition
          id="HazeClear-Mamba"
          component={Mamba}
          durationInFrames={360}
          fps={30}
          width={1280}
          height={720}
        />
        <Composition
          id="HazeClear-Reconstruction"
          component={Reconstruction}
          durationInFrames={240}
          fps={30}
          width={1280}
          height={720}
        />
        <Composition
          id="HazeClear-DenRole"
          component={DenRole}
          durationInFrames={240}
          fps={30}
          width={1280}
          height={720}
        />
        <Composition
          id="HazeClear-Detail"
          component={Detail}
          durationInFrames={300}
          fps={30}
          width={1280}
          height={720}
        />
        <Composition
          id="HazeClear-DenResult"
          component={DenResult}
          durationInFrames={240}
          fps={30}
          width={1280}
          height={720}
        />
        <Composition
          id="HazeClear-Optimization"
          component={Optimization}
          durationInFrames={420}
          fps={30}
          width={1280}
          height={720}
        />
        <Composition
          id="HazeClear-Closing"
          component={Closing}
          durationInFrames={360}
          fps={30}
          width={1280}
          height={720}
        />
      </Folder>
      <MyComposition />
    </>
  );
};
