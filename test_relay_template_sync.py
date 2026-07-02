import re
import unittest
from pathlib import Path


class RelayTemplateSyncTest(unittest.TestCase):
    """The relay page exists twice: as fluxrt_tcp_relay.html and pasted
    into RELAY_HTML_TEMPLATE in td_fluxrt_ext_tcp.py. Edits must land in
    both — this pins them together so they cannot silently drift."""

    def test_embedded_template_matches_standalone_file(self):
        root = Path(__file__).parent
        extension_source = (root / "td_fluxrt_ext_tcp.py").read_text()
        match = re.search(
            r"RELAY_HTML_TEMPLATE = '''(.*?)'''", extension_source, re.S
        )
        self.assertIsNotNone(match, "RELAY_HTML_TEMPLATE not found")
        embedded = match.group(1).strip()
        standalone = (root / "fluxrt_tcp_relay.html").read_text().strip()
        self.assertEqual(embedded, standalone)


if __name__ == "__main__":
    unittest.main()
