import {
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Stage, Pipeline, clamp, colors } from "./Visuals";

const RoleCard = ({
  left,
  top,
  title,
  text,
  delay,
}: {
  left: number;
  top: number;
  title: string;
  text: string;
  delay: number;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width: 315,
        minHeight: 118,
        padding: "20px 22px",
        boxSizing: "border-box",
        background: "#FFFFFFB8",
        borderLeft: `5px solid ${colors.red}`,
        opacity: interpolate(frame, [delay * fps, (delay + 0.6) * fps], [0, 1], clamp),
        translate: interpolate(frame, [delay * fps, (delay + 0.6) * fps], ["0px 18px", "0px 0px"], clamp),
      }}
    >
      <div style={{ fontSize: 25, fontWeight: 700, color: colors.ink }}>{title}</div>
      <div style={{ fontSize: 21, lineHeight: 1.3, marginTop: 9, color: colors.muted }}>{text}</div>
    </div>
  );
};

const DenRoleInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <Stage
      chapter="05 / VAI TRÒ CỦA DEN"
      title="DEN tinh chỉnh ảnh sau IWT"
      subtitle="DEN kết hợp xử lý không gian, tần số và kết nối dày."
    >
      <div style={{ position: "absolute", left: 70, top: 230 }}>
        <Pipeline active={4} refinement="DEN" />
      </div>
      <div
        style={{
          position: "absolute",
          left: 70,
          top: 330,
          width: 1130,
          height: 6,
          background: "#CAD9D9",
        }}
      >
        <div
          style={{
            width: interpolate(frame, [0.5 * fps, 3 * fps], [0, 100], clamp) + "%",
            height: "100%",
            background: colors.red,
          }}
        />
      </div>
      <div style={{ position: "absolute", left: 70, top: 365, fontSize: 25, color: colors.muted }}>
        Ảnh thô từ IWT
      </div>
      <div style={{ position: "absolute", left: 1040, top: 365, fontSize: 25, color: colors.muted }}>
        Ảnh đã tinh chỉnh
      </div>
      <RoleCard left={90} top={440} title="Xử lý không gian" text="Tinh chỉnh biên và chi tiết cục bộ bằng tích chập." delay={1.2} />
      <RoleCard left={480} top={440} title="Xử lý tần số" text="Điều chỉnh thành phần tần số bằng biến đổi Fourier." delay={1.7} />
      <RoleCard left={870} top={440} title="Kết nối dày" text="Tái sử dụng đặc trưng giữa các lớp xử lý." delay={2.2} />
    </Stage>
  );
};

export const DenRole = Interactive.withSchema({
  Component: DenRoleInner,
  componentName: "HazeClearDenRole",
  schema: {},
  wrapInSequence: true,
  layout: "absolute-fill",
});
