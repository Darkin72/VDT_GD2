import type { CSSProperties, ReactNode } from "react";
import {
  AbsoluteFill,
  CanvasImage,
  Interactive,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export const colors = {
  ink: "#183544",
  red: "#C52543",
  teal: "#008D87",
  cream: "#F5F4EE",
  muted: "#59717C",
};
export const clamp = {
  extrapolateLeft: "clamp",
  extrapolateRight: "clamp",
} as const;

export const Stage = ({
  chapter,
  title,
  subtitle,
  children,
  note,
}: {
  chapter: string;
  title: string;
  subtitle: string;
  children: ReactNode;
  note?: ReactNode;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <AbsoluteFill
      style={{
        fontFamily: "Calibri, sans-serif",
        color: colors.ink,
        background:
          "radial-gradient(ellipse at 88% 20%, #DCEDE9 0%, #F5F4EE 58%)",
      }}
    >
      <div
        style={{
          position: "absolute",
          left: 58,
          top: 29,
          fontSize: 19,
          letterSpacing: 3,
          fontWeight: 700,
          color: colors.red,
        }}
      >
        VIETTEL HAZECLEAR / {chapter}
      </div>
      <Interactive.Div
        name="Tiêu đề"
        style={{
          position: "absolute",
          left: 56,
          top: 78,
          right: 50,
          fontSize: 56,
          lineHeight: 1.08,
          fontWeight: 700,
          opacity: interpolate(frame, [0, 0.5 * fps], [0, 1], clamp),
          translate: interpolate(frame, [0, 0.7 * fps], ["0px 15px", "0px 0px"], clamp)
        }}
        from={57}
        trimBefore={57}
      >
        {title}
      </Interactive.Div>
      <div
        style={{
          position: "absolute",
          left: 59,
          top: 153,
          right: 50,
          fontSize: 26,
          color: colors.muted,
        }}
      >
        {subtitle}
      </div>
      {children}
      <div
        style={{
          position: "absolute",
          left: 58,
          bottom: 27,
          fontSize: 17,
          color: colors.muted,
        }}
      >
        {note === undefined
          ? "Minh họa nguyên lý • Kiến trúc kế thừa WDMamba"
          : note}
      </div>
    </AbsoluteFill>
  );
};

export const Picture = ({
  name,
  style,
}: {
  name: string;
  style?: CSSProperties;
}) => {
  const { fps } = useVideoConfig();
  return (
    <CanvasImage
      src={staticFile(`hazeclear/${name}.png`)}
      premountFor={fps}
      style={{ width: "100%", height: "100%", objectFit: "cover", ...style }}
    />
  );
};

export const Wipe = ({
  start = 1,
  end = 5,
  width = 640,
  height = 360,
}: {
  start?: number;
  end?: number;
  width?: number;
  height?: number;
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = interpolate(
    frame,
    [start * fps, end * fps],
    [0, 100],
    clamp,
  );
  return (
    <div
      style={{
        position: "relative",
        width,
        height,
        overflow: "hidden",
        borderRadius: 8,
      }}
    >
      <Picture name="hazy" />
      <div
        style={{
          position: "absolute",
          inset: 0,
          clipPath: `inset(0 ${100 - progress}% 0 0)`,
        }}
      >
        <Picture name="clear" />
      </div>
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `${progress}%`,
          width: 4,
          background: colors.red,
        }}
      />
    </div>
  );
};

export const Pipeline = ({ active = 0 }: { active?: number }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 13 }}>
    {["Ảnh sương", "DWT", "LFRN", "IWT", "DSRB", "Ảnh rõ"].map((label, i) => (
      <div
        key={label}
        style={{ display: "flex", alignItems: "center", gap: 13 }}
      >
        <div
          style={{
            padding: "18px 21px",
            borderBottom: `4px solid ${i === active ? colors.red : "#CAD9D9"}`,
            color: i === active ? colors.red : colors.muted,
            fontSize: 25,
            fontWeight: 700,
          }}
        >
          {label}
        </div>
        {i < 5 ? (
          <span style={{ fontSize: 27, color: "#91A5AB" }}>→</span>
        ) : null}
      </div>
    ))}
  </div>
);
