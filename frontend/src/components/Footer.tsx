export function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="admin-footer">
      <span>Dental Price Matcher · Admin Portal</span>
      <span>© {year}</span>
    </footer>
  );
}