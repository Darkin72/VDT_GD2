import { useVideoConfig } from "remotion";
import { Opening } from "./Opening";
import { Wavelet } from "./Wavelet";
import { Mamba } from "./Mamba";
import { Reconstruction } from "./Reconstruction";
import { Detail } from "./Detail";
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
      <Detail
        name="05 - DEN là điểm nghẽn"
        from={38 * fps}
        durationInFrames={10 * fps}
        premountFor={fps}
      />
      <Optimization
        name="06 - DSRB thay DEN"
        from={48 * fps}
        durationInFrames={14 * fps}
        premountFor={fps}
      />
      <Closing
        name="07 - Kết quả và hướng phát triển"
        from={62 * fps}
        durationInFrames={12 * fps}
        premountFor={fps}
      />
    </>
  );
};
