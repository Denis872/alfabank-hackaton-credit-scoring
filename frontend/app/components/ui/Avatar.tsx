export function Avatar({
  src,
  alt,
  size = 48,
}: {
  src?: string;
  alt?: string;
  size?: number;
}) {
  return (
    <img
      src={src ?? "/avatar.jpg"}
      alt={alt ?? "avatar"}
      style={{ width: size, height: size }}
      className="rounded-full object-cover"
    />
  );
}
