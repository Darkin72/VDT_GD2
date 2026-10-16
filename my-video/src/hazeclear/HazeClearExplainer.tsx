import { useVideoConfig } from "remotion";
import { Opening } from "./Opening";
import { Wavelet } from "./Wavelet";
import { Mamba } from "./Mamba";
import { Reconstruction } from "./Reconstruction";
import { DenRole } from "./DenRole";
import { Detail } from "./Detail";
import { DenResult } from "./DenResult";
import { Optimization } from "./Optimization";
import { Closing } from "./Closing";

export const HazeClearExplainer = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Opening
        name="01 - Ảnh có sương"
        from={0}
        durationInFrames={8 * fps}
        premountFor={fps}
      />
      <Wavelet
        name="02 - Tách Wavelet"
        from={8 * fps}
        durationInFrames={10 * fps}
        premountFor={fps}
      />
      <Mamba
        name="03 - Mamba quét bốn hướng"
        from={18 * fps}
        durationInFrames={12 * fps}
        premountFor={fps}
      />
      <Reconstruction
        name="04 - Tái dựng IWT"
        from={30 * fps}
        durationInFrames={8 * fps}
        premountFor={fps}
      />
      <DenRole
        name="05 - Vai trò của DEN"
        from={38 * fps}
        durationInFrames={8 * fps}
        premountFor={fps}
      />
      <Detail
        name="06 - Đo thời gian DEN"
        from={46 * fps}
        durationInFrames={10 * fps}
        premountFor={fps}
      />
      <DenResult
        name="07 - Kết quả DEN"
        from={56 * fps}
        durationInFrames={8 * fps}
        premountFor={fps}
      />
      <Optimization
        name="08 - DSRB thay DEN"
        from={64 * fps}
        durationInFrames={14 * fps}
        premountFor={fps}
      />
      <Closing
        name="09 - Kết quả và hướng phát triển"
        from={78 * fps}
        durationInFrames={12 * fps}
        premountFor={fps}
      />
    </>
  );
};
