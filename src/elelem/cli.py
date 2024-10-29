# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: AGPL-3.0-or-later

import click

from . import plugins


class LazyProviderGroup(click.Group):
    def list_commands(self, ctx: click.Context):
        base = super().list_commands(ctx)
        providers = plugins.load_providers()
        return base + sorted(providers.keys())

    def get_command(self, ctx: click.Context, cmd_name: str):
        return plugins.load_providers()[cmd_name]


@click.group(cls=LazyProviderGroup)
# XXX global options, location of .env etc.
@click.version_option()
def cli():
    pass
